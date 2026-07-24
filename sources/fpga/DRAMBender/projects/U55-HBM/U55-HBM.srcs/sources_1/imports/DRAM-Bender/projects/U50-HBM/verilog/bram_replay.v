`include "parameters.vh"
`include "project.vh"

// ===========================================================================
// bram_replay : on-chip command-trace replay engine
// ===========================================================================
//
// Loads a *compressed* HBM command trace into on-chip memory (inferred BRAM/
// URAM, no IP core required) and replays it in a hardware loop with ZERO host
// / PCIe involvement during the run.  Companion to stream_test.v, which streams
// the (NOP-materialised) command list from the host every loop iteration; this
// module instead stores only the non-NOP commands and reconstructs the timing
// from a free-running cycle counter.
//
// == Timing model (4 slots / fabric cycle, == stream_test rate) ==
// The downstream HBM pipeline consumes 4 command slots per fab_clk cycle
// (150 MHz x 4 = 600 M slots/s).  A single 64-bit slot counter `cyc` advances
// by 4 every fab_clk; lane L (0..3) corresponds to slot (cyc + L).  The
// compressed trace is split host-side into 4 banks by (timestamp mod 4); bank L
// therefore only ever contains commands whose slot index lands on lane L.  Each
// bank has its own comparator: when the head entry's timestamp == cyc + L, that
// command is issued on lane L and the entry is dequeued; otherwise lane L = NOP.
// This reproduces the exact command schedule (in slot units) of stream mode.
//
// == Compressed entry ==
//   entry[63:32] = timestamp (slot index, absolute within one iteration)
//   entry[31: 0] = 32-bit encoded command (same layout as stream_test /
//                  platform.cpp encodeCommand)
//
// == Host control protocol (over the H2C AXI-Stream, while bram_replay_active) ==
// Each 256-bit beat carries a type tag in byte 31 (h2c_tdata[255:248]):
//   TYPE_CONFIG (0): [31:0]=trace_len_slots  [63:32]=num_iters (0 => infinite)
//                    [160]=wr_data_enable    [223:192]=wr_pattern (32b, x16)
//   TYPE_ENTRY  (1): [31:0]=command          [63:32]=timestamp
//                    routed to bank = timestamp[1:0]
//   TYPE_DONE   (2): end of load -> latch per-bank counts, prime, then run
//   TYPE_STOP   (3): request graceful stop (finish current iteration, then idle)
// Byte 8 (h2c_tdata[71:64]) is left 0 by the host so it never collides with the
// frontend reset bit (h2c_tdata[`INSTR_WIDTH]) which stays live in this mode.
//
// == Result ==
// On completion (num_iters reached, or stop honoured at an iteration boundary)
// pulses bram_replay_done for one cycle with bram_replay_cycle_count = total
// fab_clk cycles spent replaying (excludes load/prime), reusing the same C2H
// readback path as stream_test (softmc_top muxes done/count into the readback
// engine, host parses {8x count}).
// ===========================================================================

module bram_replay #(
  parameter integer BANK_DEPTH = 131072,          // entries per bank (>= max non-NOP cmds/4);
                                                   //   131072 holds the full 1-rank A100 HBM2
                                                   //   per-PC traces (~69k/bank) with headroom.
                                                   //   ~32 URAMs/bank x 4 banks = ~128 URAMs.
  parameter integer PF_DEPTH   = 8,               // per-bank prefetch FIFO depth (power of 2, >=2)
  parameter integer PRIME_CYCLES = 16             // warm-up cycles to fill prefetch FIFOs
)(
  input                                   clk,
  input                                   rst,

  // Control (asserted by frontend while in BRAM_REPLAY_S)
  input                                   bram_replay_active,

  // AXI-Stream H2C input (fab_clk domain, 256-bit)
  input  [`XDMA_AXI_DATA_WIDTH-1:0]       h2c_tdata,
  input                                   h2c_tvalid,
  output                                  h2c_tready,

  // HBM command outputs (4 lanes), mirrors stream_test's st_ddr_* port group
  output reg [3:0]                        br_ddr_act,
  output reg [3:0]                        br_ddr_pre,
  output reg [3:0]                        br_ddr_read,
  output reg [3:0]                        br_ddr_write,
  output reg [3:0]                        br_ddr_ref,
  output reg [3:0]                        br_ddr_nop,
  output reg [3:0]                        br_ddr_ap,
  output reg [3:0]                        br_ddr_pall,
  output reg [3:0]                        br_ddr_rank,
  output reg [3:0]                        br_hbm_sel_ch,
  output reg [4*`HBM_CH_WIDTH-1:0]        br_hbm_ch,
  output reg [4*`BG_WIDTH-1:0]            br_ddr_bg,
  output reg [4*`BANK_WIDTH-1:0]          br_ddr_bank,
  output reg [4*`COL_WIDTH-1:0]           br_ddr_col,
  output reg [4*`ROW_WIDTH-1:0]           br_ddr_row,
  output reg [511:0]                      br_ddr_wdata,
  output reg                              cmd_valid,

  // Result
  output reg                              bram_replay_done,
  output reg [63:0]                       bram_replay_cycle_count
);

  localparam integer ADDR_W = (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
  localparam integer PF_W   = (PF_DEPTH   <= 1) ? 1 : $clog2(PF_DEPTH);

  // Beat type tags (byte 31 of the 256-bit H2C word)
  localparam [7:0] TYPE_CONFIG = 8'd0,
                   TYPE_ENTRY  = 8'd1,
                   TYPE_DONE   = 8'd2,
                   TYPE_STOP   = 8'd3;

  // Sub-FSM
  localparam [2:0] BR_IDLE = 3'd0,
                   BR_LOAD = 3'd1,
                   BR_PRIME= 3'd2,
                   BR_RUN  = 3'd3,
                   BR_DONE = 3'd4;
  reg [2:0] state_r;

  // ---------------------------------------------------------------
  // H2C beat decode
  // ---------------------------------------------------------------
  wire [7:0]  beat_type   = h2c_tdata[255:248];
  wire [31:0] beat_lo     = h2c_tdata[31:0];
  wire [31:0] beat_hi     = h2c_tdata[63:32];
  wire [31:0] cfg_wr_pat  = h2c_tdata[223:192];
  wire        cfg_wr_en   = h2c_tdata[160];
  wire [1:0]  entry_bank  = beat_hi[1:0];            // timestamp mod 4

  // Accept beats during LOAD (store) and RUN (only to catch STOP)
  assign h2c_tready = (state_r == BR_LOAD) | (state_r == BR_RUN);
  wire   beat       = h2c_tvalid & h2c_tready;

  // ---------------------------------------------------------------
  // Config / control registers
  // ---------------------------------------------------------------
  reg [31:0] trace_len;          // slots in one iteration
  reg [31:0] num_iters;          // 0 => infinite
  reg [31:0] wr_pattern;
  reg        wr_data_en;
  reg        stop_req;

  // ---------------------------------------------------------------
  // Global replay counters
  // ---------------------------------------------------------------
  reg [63:0] cyc;                // slot index of lane 0 this fab cycle (multiple of 4)
  reg [31:0] iter;              // completed iterations
  reg [31:0] prime_ctr;
  reg        iter_boundary;     // pulse: end of an iteration this cycle

  wire running = (state_r == BR_RUN);

  // ---------------------------------------------------------------
  // Per-bank storage + prefetch FIFO (one per lane)
  // ---------------------------------------------------------------
  // Aggregated per-lane outputs from the generate block
  wire [31:0] lane_cmd  [0:3];
  wire        lane_fire [0:3];   // this lane issues a command this cycle
  wire [3:0]  banks_primed;      // each bank has primed its FIFO (or is empty)

  genvar L;
  generate
    for (L = 0; L < 4; L = L + 1) begin : gb
      // --- trace storage: inferred large memory (URAM-friendly) ---
      (* ram_style = "ultra" *) reg [63:0] mem [0:BANK_DEPTH-1];

      reg  [ADDR_W-1:0] wr_ptr;            // load write pointer
      reg  [ADDR_W-1:0] cnt;               // #entries (latched at TYPE_DONE)
      reg  [ADDR_W-1:0] rd_ptr;            // next entry to prefetch
      reg  [63:0]       mem_dout;          // registered read data
      reg               rd_issued_r;       // a read was issued last cycle -> push now

      // --- load write (route by timestamp mod 4) ---
      wire do_wr = beat & (beat_type == TYPE_ENTRY) & (entry_bank == L[1:0]);

      // --- prefetch FIFO (small, register/LUTRAM based) ---
      reg  [63:0]      pf [0:PF_DEPTH-1];
      reg  [PF_W-1:0]  pf_head, pf_tail;
      reg  [PF_W:0]    pf_count;           // 0..PF_DEPTH

      wire pf_full   = (pf_count == PF_DEPTH[PF_W:0]);
      wire pf_empty  = (pf_count == 0);
      wire more_mem  = (rd_ptr < cnt);     // entries left in this bank's memory

      // Issue a memory read when there's room (accounting for the in-flight one)
      // and entries remain.  pop<=1/cyc, refill=1/cyc => FIFO never underflows
      // once primed (PF_DEPTH >= read latency + 1 = 2).
      wire room_for_read = ((pf_count + {{(PF_W){1'b0}}, rd_issued_r}) < PF_DEPTH[PF_W:0]);
      wire do_read = (state_r == BR_PRIME || state_r == BR_RUN) & more_mem & room_for_read;

      // Head of FIFO
      wire [63:0] head      = pf[pf_head];
      wire [31:0] head_ts   = head[63:32];
      wire [31:0] head_cmd  = head[31:0];
      wire        head_valid= ~pf_empty;

      // Match: this lane's slot (cyc + L) equals the head timestamp
      wire match = running & head_valid & (head_ts == (cyc[31:0] + L[31:0]));

      assign lane_cmd[L]  = head_cmd;
      assign lane_fire[L] = match;

      // Primed when FIFO has data or the bank is empty (nothing to wait for)
      assign banks_primed[L] = (~pf_empty) | (cnt == 0);

      // push-from-memory this cycle (result of read issued last cycle)
      wire pf_push = rd_issued_r;
      wire pf_pop  = match;

      integer k;
      always @(posedge clk) begin
        if (rst | ~bram_replay_active | (state_r == BR_IDLE)) begin
          wr_ptr      <= {ADDR_W{1'b0}};
          cnt         <= {ADDR_W{1'b0}};
          rd_ptr      <= {ADDR_W{1'b0}};
          rd_issued_r <= 1'b0;
          pf_head     <= {PF_W{1'b0}};
          pf_tail     <= {PF_W{1'b0}};
          pf_count    <= {(PF_W+1){1'b0}};
        end
        else begin
          // ---- load phase: write entries, latch count on DONE ----
          if (do_wr) begin
            mem[wr_ptr] <= {beat_hi, beat_lo};   // {ts, cmd}
            wr_ptr      <= wr_ptr + 1'b1;
          end
          if (beat & (beat_type == TYPE_DONE))
            cnt <= wr_ptr;

          // ---- reset read side at start of each iteration / prime ----
          if (iter_boundary | (state_r == BR_DONE)) begin
            rd_ptr      <= {ADDR_W{1'b0}};
            rd_issued_r <= 1'b0;
            pf_head     <= {PF_W{1'b0}};
            pf_tail     <= {PF_W{1'b0}};
            pf_count    <= {(PF_W+1){1'b0}};
          end
          else begin
            // memory read pipeline (1-cycle latency)
            if (do_read) begin
              mem_dout    <= mem[rd_ptr];
              rd_ptr      <= rd_ptr + 1'b1;
              rd_issued_r <= 1'b1;
            end
            else begin
              rd_issued_r <= 1'b0;
            end

            // FIFO push (from completed read) / pop (on match).
            // pf_head/pf_tail are PF_W bits and wrap naturally (PF_DEPTH is a
            // power of two — see parameter note).
            if (pf_push) begin
              pf[pf_tail] <= mem_dout;
              pf_tail     <= pf_tail + 1'b1;
            end
            if (pf_pop) begin
              pf_head     <= pf_head + 1'b1;
            end
            case ({pf_push, pf_pop})
              2'b10:   pf_count <= pf_count + 1'b1;
              2'b01:   pf_count <= pf_count - 1'b1;
              default: pf_count <= pf_count;   // 00 or 11 : no net change
            endcase
          end
        end
      end
    end
  endgenerate

  // ---------------------------------------------------------------
  // Main control FSM + counters
  // ---------------------------------------------------------------
  // last slot index of the current fab cycle is cyc+3; an iteration of
  // trace_len slots is complete once cyc+4 >= trace_len.
  wire iter_complete = running & ((cyc + 64'd4) >= {32'd0, trace_len});
  wire all_primed    = (&banks_primed);

  always @(posedge clk) begin
    if (rst | ~bram_replay_active) begin
      state_r                 <= BR_IDLE;
      trace_len               <= 32'd0;
      num_iters               <= 32'd0;
      wr_pattern              <= 32'hDEAD_BEEF;
      wr_data_en              <= 1'b0;
      stop_req                <= 1'b0;
      cyc                     <= 64'd0;
      iter                    <= 32'd0;
      prime_ctr               <= 32'd0;
      iter_boundary           <= 1'b0;
      bram_replay_done        <= 1'b0;
      bram_replay_cycle_count <= 64'd0;
    end
    else begin
      iter_boundary    <= 1'b0;
      bram_replay_done <= 1'b0;

      case (state_r)
        BR_IDLE: begin
          // entered active: start receiving the load stream
          state_r                 <= BR_LOAD;
          cyc                     <= 64'd0;
          iter                    <= 32'd0;
          stop_req                <= 1'b0;
          bram_replay_cycle_count <= 64'd0;
        end

        BR_LOAD: begin
          if (beat) begin
            case (beat_type)
              TYPE_CONFIG: begin
                trace_len  <= beat_lo;
                num_iters  <= beat_hi;
                wr_data_en <= cfg_wr_en;
                wr_pattern <= cfg_wr_pat;
              end
              TYPE_DONE: begin
                state_r   <= BR_PRIME;
                prime_ctr <= 32'd0;
              end
              default: ; // TYPE_ENTRY handled in per-bank block; STOP ignored here
            endcase
          end
        end

        BR_PRIME: begin
          // hold cyc at 0, let prefetch FIFOs fill
          cyc <= 64'd0;
          if ((prime_ctr >= PRIME_CYCLES[31:0]) && all_primed)
            state_r <= BR_RUN;
          else
            prime_ctr <= prime_ctr + 32'd1;
        end

        BR_RUN: begin
          // catch a graceful-stop request
          if (beat & (beat_type == TYPE_STOP))
            stop_req <= 1'b1;

          // advance the slot counter (4 slots / fab cycle) and total counter
          cyc                     <= cyc + 64'd4;
          bram_replay_cycle_count <= bram_replay_cycle_count + 64'd1;

          if (iter_complete) begin
            iter_boundary <= 1'b1;          // resets per-bank read side
            cyc           <= 64'd0;
            iter          <= iter + 32'd1;
            // stop if reached requested count, or a stop was requested
            if (((num_iters != 32'd0) && ((iter + 32'd1) >= num_iters)) || stop_req)
              state_r <= BR_DONE;
            else
              state_r <= BR_PRIME;          // re-prime FIFOs for next iteration
            prime_ctr <= 32'd0;
          end
        end

        BR_DONE: begin
          bram_replay_done <= 1'b1;         // one-cycle pulse (count already final)
          // stay here until frontend drops bram_replay_active
        end

        default: state_r <= BR_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------
  // Combinational command decode (per lane), mirrors stream_test.v
  // ---------------------------------------------------------------
  localparam [511:0] DUMMY_WDATA = {16{32'hDEAD_BEEF}};

  integer i;
  reg [3:0]                  c_type;
  reg [`ROW_ADDR_WIDTH-1:0]  c_row;
  reg [`COL_ADDR_WIDTH-1:0]  c_col;
  reg [`BG_WIDTH-1:0]        c_bg;
  reg [`BANK_WIDTH-1:0]      c_bank;
  reg                        c_pc;
  reg [`HBM_CH_WIDTH-1:0]    c_ch;
  reg [31:0]                 c_word;

  always @(*) begin
    br_ddr_act    = 4'b0;
    br_ddr_pre    = 4'b0;
    br_ddr_read   = 4'b0;
    br_ddr_write  = 4'b0;
    br_ddr_ref    = 4'b0;
    br_ddr_nop    = 4'b1111;
    br_ddr_ap     = 4'b0;
    br_ddr_pall   = 4'b0;
    br_ddr_rank   = 4'b0;
    br_hbm_sel_ch = 4'b0;
    br_hbm_ch     = {(4*`HBM_CH_WIDTH){1'b0}};
    br_ddr_bg     = {(4*`BG_WIDTH){1'b0}};
    br_ddr_bank   = {(4*`BANK_WIDTH){1'b0}};
    br_ddr_col    = {(4*`COL_WIDTH){1'b0}};
    br_ddr_row    = {(4*`ROW_WIDTH){1'b0}};
    // Fixed write pattern (replicated across both PC halves). Per-command write
    // data is not stored in v1; WR/WRA use wr_pattern when wr_data_en, else dummy.
    br_ddr_wdata  = wr_data_en ? {16{wr_pattern}} : DUMMY_WDATA;
    cmd_valid     = 1'b0;

    for (i = 0; i < 4; i = i + 1) begin
      c_word = lane_cmd[i];
      c_type = c_word[31:28];
      c_row  = c_word[27:14];
      c_col  = c_word[13:9];
      c_bg   = c_word[8:7];
      c_bank = c_word[6:5];
      c_pc   = c_word[4];
      c_ch   = c_word[3:0];

      if (lane_fire[i]) begin
        cmd_valid = 1'b1;
        br_ddr_row [`ROW_WIDTH*i  +: `ROW_WIDTH]  = {{(`ROW_WIDTH-`ROW_ADDR_WIDTH){1'b0}}, c_row};
        br_ddr_col [`COL_WIDTH*i  +: `COL_WIDTH]  = {{(`COL_WIDTH-`COL_ADDR_WIDTH){1'b0}}, c_col};
        br_ddr_bg  [`BG_WIDTH*i   +: `BG_WIDTH]   = c_bg;
        br_ddr_bank[`BANK_WIDTH*i +: `BANK_WIDTH] = c_bank;
        br_ddr_rank[i]                            = c_pc;
        br_hbm_ch  [`HBM_CH_WIDTH*i +: `HBM_CH_WIDTH] = c_ch;
        br_hbm_sel_ch[i]                          = 1'b1;

        br_ddr_nop[i] = 1'b0;
        case (c_type)
          `ACTT:  br_ddr_act[i]   = 1'b1;
          `PREE:  br_ddr_pre[i]   = 1'b1;
          `PREA:  begin br_ddr_pre[i] = 1'b1; br_ddr_pall[i] = 1'b1; end
          `RD:    br_ddr_read[i]  = 1'b1;
          `RDA:   begin br_ddr_read[i] = 1'b1; br_ddr_ap[i] = 1'b1; end
          `WR:    br_ddr_write[i] = 1'b1;
          `WRA:   begin br_ddr_write[i] = 1'b1; br_ddr_ap[i] = 1'b1; end
          `REFF:  br_ddr_ref[i]   = 1'b1;
          default:br_ddr_nop[i]   = 1'b1;
        endcase
      end
    end
  end

endmodule
