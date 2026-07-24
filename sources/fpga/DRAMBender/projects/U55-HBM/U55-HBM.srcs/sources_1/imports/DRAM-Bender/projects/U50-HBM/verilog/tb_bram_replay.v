`timescale 1ns/1ps
`include "parameters.vh"
`include "project.vh"

// ===========================================================================
// tb_bram_replay : self-checking testbench for bram_replay.v
// ===========================================================================
//
// Drives the SAME H2C beat protocol the host (platform.cpp loadTraceAndRun)
// produces, builds a reference model of the trace, and verifies on every cycle:
//   * which lane fires, the command type, and the slot index (== cyc + lane)
//   * exactly the expected commands issue, none extra, none duplicated
//   * each iteration issues the full command set
//   * the loop runs the requested number of iterations
//   * the returned cycle count == num_iters * ceil(trace_len/4)
//   * the per-bank prefetch FIFO never underflows on dense back-to-back hits
//   * graceful stop finishes the current iteration then halts (no partial iter)
//
// Run:
//   iverilog -g2012 \
//     -I <...>/sources/hdl/header_verilog -I <...>/projects/U50-HBM/verilog \
//     tb_bram_replay.v bram_replay.v -o /tmp/tb_bram_replay && vvp /tmp/tb_bram_replay
// ===========================================================================

module tb_bram_replay;

  // ---- DUT params (small + fast for sim) ----
  localparam integer BANK_DEPTH   = 4096;
  localparam integer PF_DEPTH     = 8;
  localparam integer PRIME_CYCLES = 4;

  // DUT sub-FSM state encodings (must match bram_replay.v)
  localparam [2:0] S_IDLE = 3'd0, S_LOAD = 3'd1, S_PRIME = 3'd2, S_RUN = 3'd3, S_DONE = 3'd4;

  // ---- clock / reset / control ----
  reg clk = 1'b0;
  reg rst = 1'b1;
  reg active = 1'b0;
  always #5 clk = ~clk;   // 100 MHz sim clock (rate irrelevant)

  // ---- H2C stream ----
  reg  [`XDMA_AXI_DATA_WIDTH-1:0] h2c_tdata;
  reg                             h2c_tvalid;
  wire                            h2c_tready;

  // ---- HBM command outputs ----
  wire [3:0] act, pre, rd, wr, refc, nop, ap, pall, rank, sel;
  wire [4*`HBM_CH_WIDTH-1:0] hch;
  wire [4*`BG_WIDTH-1:0]     bg;
  wire [4*`BANK_WIDTH-1:0]   bk;
  wire [4*`COL_WIDTH-1:0]    col;
  wire [4*`ROW_WIDTH-1:0]    row;
  wire [511:0]               wd;
  wire                       cv;
  wire                       done;
  wire [63:0]                cyccount;

  bram_replay #(.BANK_DEPTH(BANK_DEPTH), .PF_DEPTH(PF_DEPTH), .PRIME_CYCLES(PRIME_CYCLES)) dut (
    .clk(clk), .rst(rst), .bram_replay_active(active),
    .h2c_tdata(h2c_tdata), .h2c_tvalid(h2c_tvalid), .h2c_tready(h2c_tready),
    .br_ddr_act(act), .br_ddr_pre(pre), .br_ddr_read(rd), .br_ddr_write(wr),
    .br_ddr_ref(refc), .br_ddr_nop(nop), .br_ddr_ap(ap), .br_ddr_pall(pall),
    .br_ddr_rank(rank), .br_hbm_sel_ch(sel), .br_hbm_ch(hch),
    .br_ddr_bg(bg), .br_ddr_bank(bk), .br_ddr_col(col), .br_ddr_row(row),
    .br_ddr_wdata(wd), .cmd_valid(cv),
    .bram_replay_done(done), .bram_replay_cycle_count(cyccount)
  );

  // ---- command encode (matches platform.cpp encodeCommand) ----
  function [31:0] enc(input [3:0] t, input [13:0] r, input [4:0] c,
                      input [1:0] g, input [1:0] b, input p, input [3:0] ch);
    enc = ({28'b0, t} << 28) | ({18'b0, r} << 14) | ({27'b0, c} << 9)
        | ({30'b0, g} << 7) | ({30'b0, b} << 5) | ({31'b0, p} << 4) | {28'b0, ch};
  endfunction

  // ---- beat builders (match the host byte/bit layout) ----
  function [255:0] config_beat(input [31:0] tlen, input [31:0] niters,
                               input wr_en, input [31:0] wpat);
    begin
      config_beat            = 256'b0;
      config_beat[31:0]      = tlen;
      config_beat[63:32]     = niters;
      config_beat[160]       = wr_en;
      config_beat[223:192]   = wpat;
      config_beat[255:248]   = 8'd0;   // TYPE_CONFIG
    end
  endfunction
  function [255:0] entry_beat(input [31:0] ts, input [31:0] cmd);
    begin
      entry_beat           = 256'b0;
      entry_beat[31:0]     = cmd;
      entry_beat[63:32]    = ts;
      entry_beat[255:248]  = 8'd1;     // TYPE_ENTRY
    end
  endfunction
  function [255:0] done_beat(input dummy); begin done_beat = 256'b0; done_beat[255:248] = 8'd2; end endfunction
  function [255:0] stop_beat(input dummy); begin stop_beat = 256'b0; stop_beat[255:248] = 8'd3; end endfunction

  // ---- send one beat respecting tready ----
  task send_beat(input [255:0] data);
    begin
      @(negedge clk);
      while (!h2c_tready) @(negedge clk);   // state is stable mid-cycle
      h2c_tdata  = data;
      h2c_tvalid = 1'b1;
      @(posedge clk);                       // accepted here (tvalid & tready)
      @(negedge clk);
      h2c_tvalid = 1'b0;
    end
  endtask

  // ---- reference model ----
  localparam integer MAXSLOT = 1024;
  reg        ref_valid [0:MAXSLOT-1];
  reg [3:0]  ref_type  [0:MAXSLOT-1];
  reg        seen      [0:MAXSLOT-1];
  integer    n_entries_tb;
  integer    trace_len_tb;
  integer    errors;
  integer    fires_this_iter;
  integer    total_fires;
  integer    prev_iter;
  integer    iters_done3;
  reg        checking;

  integer ci;
  task reset_refs;
    begin
      for (ci = 0; ci < MAXSLOT; ci = ci + 1) begin
        ref_valid[ci] = 1'b0; ref_type[ci] = 4'd0; seen[ci] = 1'b0;
      end
      n_entries_tb = 0; fires_this_iter = 0; total_fires = 0; prev_iter = 0;
    end
  endtask

  // load one entry: set reference AND stream the ENTRY beat
  task load_entry(input [31:0] ts, input [31:0] cmd, input [3:0] typ);
    begin
      ref_valid[ts] = 1'b1;
      ref_type[ts]  = typ;
      n_entries_tb  = n_entries_tb + 1;
      send_beat(entry_beat(ts, cmd));
    end
  endtask

  // decode the fired command type on lane i from the output strobes
  function [3:0] lane_type(input integer i);
    begin
      if      (act[i])              lane_type = `ACTT;
      else if (pre[i] && pall[i])   lane_type = `PREA;
      else if (pre[i])              lane_type = `PREE;
      else if (rd[i] && ap[i])      lane_type = `RDA;
      else if (rd[i])               lane_type = `RD;
      else if (wr[i] && ap[i])      lane_type = `WRA;
      else if (wr[i])               lane_type = `WR;
      else if (refc[i])             lane_type = `REFF;
      else                          lane_type = `RNOP;
    end
  endfunction

  // ---- per-cycle checker (iteration boundary first, then fire recording) ----
  integer li, kk, slot;
  always @(posedge clk) begin
    if (checking) begin
      // iteration boundary: dut.iter advanced -> the previous iteration just ended
      if (dut.iter !== prev_iter) begin
        if (fires_this_iter !== n_entries_tb) begin
          $display("[%0t] ERROR iter %0d issued %0d cmds, expected %0d",
                   $time, prev_iter, fires_this_iter, n_entries_tb);
          errors = errors + 1;
        end
        fires_this_iter = 0;
        for (kk = 0; kk < MAXSLOT; kk = kk + 1) seen[kk] = 1'b0;
        prev_iter = dut.iter;
      end
      // fire recording (only in RUN)
      if (dut.state_r == S_RUN) begin
        for (li = 0; li < 4; li = li + 1) begin
          if (!nop[li]) begin
            slot = dut.cyc[31:0] + li;
            total_fires = total_fires + 1;
            if (slot >= trace_len_tb) begin
              $display("[%0t] ERROR fire beyond trace_len at slot %0d (lane %0d)", $time, slot, li);
              errors = errors + 1;
            end else if (!ref_valid[slot]) begin
              $display("[%0t] ERROR unexpected fire at slot %0d (lane %0d, iter %0d)",
                       $time, slot, li, dut.iter);
              errors = errors + 1;
            end else begin
              if (lane_type(li) !== ref_type[slot]) begin
                $display("[%0t] ERROR type mismatch at slot %0d: exp %0d got %0d",
                         $time, slot, ref_type[slot], lane_type(li));
                errors = errors + 1;
              end
              if (seen[slot]) begin
                $display("[%0t] ERROR duplicate fire at slot %0d", $time, slot);
                errors = errors + 1;
              end
              seen[slot] = 1'b1;
              fires_this_iter = fires_this_iter + 1;
            end
          end
        end
      end
    end
  end

  // ---- helpers to bound the run ----
  task wait_done(input integer timeout);
    integer t;
    begin
      t = 0;
      while (done !== 1'b1 && t < timeout) begin @(posedge clk); t = t + 1; end
      if (done !== 1'b1) begin
        $display("[%0t] ERROR timeout waiting for done", $time);
        errors = errors + 1;
      end
    end
  endtask

  task expect_eq(input [63:0] got, input [63:0] exp, input [127:0] name);
    begin
      if (got !== exp) begin
        $display("[%0t] ERROR %0s: got %0d expected %0d", $time, name, got, exp);
        errors = errors + 1;
      end
    end
  endtask

  // =========================================================================
  // Tests
  // =========================================================================
  integer e0;
  initial begin
    $dumpfile("tb_bram_replay.vcd");
    $dumpvars(0, tb_bram_replay);
    errors = 0;
    h2c_tvalid = 1'b0; h2c_tdata = 256'b0;
    rst = 1'b1; active = 1'b0;
    repeat (5) @(posedge clk);
    rst = 1'b0;
    @(negedge clk);

    // ---------------------------------------------------------------------
    // TEST 1: multi-lane trace, finite iterations
    //   commands across all 4 lanes + gaps; 3 iterations
    // ---------------------------------------------------------------------
    $display("\n=== TEST 1: multi-lane, 3 iterations ===");
    e0 = errors;
    reset_refs;
    trace_len_tb = 24;
    active = 1'b1;
    send_beat(config_beat(24, 3, 1'b0, 32'hDEADBEEF));
    load_entry(1,  enc(`ACTT,14'd5,5'd0,2'd0,2'd1,1'b0,4'd8), `ACTT); // lane1
    load_entry(3,  enc(`ACTT,14'd6,5'd0,2'd1,2'd0,1'b1,4'd9), `ACTT); // lane3
    load_entry(4,  enc(`RD,  14'd5,5'd2,2'd0,2'd1,1'b0,4'd8), `RD);   // lane0
    load_entry(10, enc(`RD,  14'd6,5'd3,2'd1,2'd0,1'b1,4'd9), `RD);   // lane2
    load_entry(11, enc(`WR,  14'd7,5'd4,2'd2,2'd2,1'b0,4'd10),`WR);   // lane3
    load_entry(20, enc(`PREE,14'd5,5'd0,2'd0,2'd1,1'b0,4'd8), `PREE); // lane0
    checking = 1'b1; prev_iter = 0;
    send_beat(done_beat(1'b0));
    wait_done(2000);
    @(posedge clk); checking = 1'b0;
    expect_eq(total_fires, 3*6, "test1 total fires");
    expect_eq(cyccount, 3*(24/4), "test1 cycle count");
    active = 1'b0; repeat (6) @(posedge clk);
    $display("TEST 1: %0s", (errors==e0) ? "PASS" : "FAIL");

    // ---------------------------------------------------------------------
    // TEST 2: prefetch stress -- 20 commands all in ONE bank, spaced 4 slots
    //   (back-to-back hits on a single lane every fab cycle; tests no-underflow)
    // ---------------------------------------------------------------------
    $display("\n=== TEST 2: prefetch stress (20 back-to-back, single bank) ===");
    e0 = errors;
    reset_refs;
    trace_len_tb = 80;
    active = 1'b1;
    send_beat(config_beat(80, 2, 1'b0, 32'h0));
    for (kk = 0; kk < 20; kk = kk + 1)
      load_entry(kk*4, enc(`RD,14'd0,5'd0,2'd0,2'd0,1'b0,4'd8), `RD); // all ts%4==0 -> lane0
    checking = 1'b1; prev_iter = 0;
    send_beat(done_beat(1'b0));
    wait_done(4000);
    @(posedge clk); checking = 1'b0;
    expect_eq(total_fires, 2*20, "test2 total fires (no underflow)");
    expect_eq(cyccount, 2*(80/4), "test2 cycle count");
    active = 1'b0; repeat (6) @(posedge clk);
    $display("TEST 2: %0s", (errors==e0) ? "PASS" : "FAIL");

    // ---------------------------------------------------------------------
    // TEST 3: graceful stop (infinite run, stop mid-flight)
    //   must finish the current iteration then halt -> cycle count a whole
    //   multiple of cycles-per-iteration, no partial iteration.
    // ---------------------------------------------------------------------
    $display("\n=== TEST 3: graceful stop (infinite -> stop) ===");
    e0 = errors;
    reset_refs;
    trace_len_tb = 24;
    active = 1'b1;
    send_beat(config_beat(24, 0, 1'b0, 32'h0));   // num_iters = 0 => infinite
    load_entry(1,  enc(`ACTT,14'd5,5'd0,2'd0,2'd1,1'b0,4'd8), `ACTT);
    load_entry(10, enc(`RD,  14'd6,5'd3,2'd1,2'd0,1'b1,4'd9), `RD);
    load_entry(11, enc(`WR,  14'd7,5'd4,2'd2,2'd2,1'b0,4'd10),`WR);
    checking = 1'b1; prev_iter = 0;
    send_beat(done_beat(1'b0));
    // let ~2.5 iterations elapse, then request stop mid-iteration
    repeat (3 + 3*(24/4)) @(posedge clk);
    send_beat(stop_beat(1'b0));
    wait_done(3000);
    @(posedge clk); checking = 1'b0;
    iters_done3 = cyccount / (trace_len_tb/4);
    if (cyccount == 0 || (cyccount % (trace_len_tb/4)) != 0) begin
      $display("[%0t] ERROR test3 stopped mid-iteration: cyc=%0d not a whole multiple of %0d",
               $time, cyccount, trace_len_tb/4);
      errors = errors + 1;
    end
    // every completed iteration must have issued all 3 commands
    expect_eq(total_fires, iters_done3*3, "test3 fires == completed_iters*3");
    active = 1'b0; repeat (6) @(posedge clk);
    $display("TEST 3: %0s (completed %0d iterations before stop)",
             (errors==e0) ? "PASS" : "FAIL", iters_done3);

    // ---------------------------------------------------------------------
    $display("\n=========================================");
    if (errors == 0) $display("ALL TESTS PASSED");
    else             $display("FAILED: %0d error(s)", errors);
    $display("=========================================");
    $finish;
  end

  // global safety timeout
  initial begin
    #500000;
    $display("[%0t] GLOBAL TIMEOUT", $time);
    $finish;
  end

endmodule
