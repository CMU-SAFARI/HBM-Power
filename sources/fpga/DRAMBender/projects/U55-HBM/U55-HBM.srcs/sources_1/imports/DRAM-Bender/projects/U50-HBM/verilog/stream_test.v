`include "parameters.vh"
`include "project.vh"

// Stream command module with write-data support.
//
// Streams 32-bit encoded HBM commands from the host via PCIe/AXI-Stream,
// decodes them, and drives HBM adapter command signals directly —
// bypassing the normal DRAM Bender instruction pipeline.
//
// == Write-data protocol ==
// The first 256-bit AXI beat is a config header:
//   header[31:0]  = consume_divider (unused by stream_test, kept for compat)
//   header[32]    = wr_data_enable  (1 = write data beats follow WR groups)
//   header[255:33]= reserved (zero)
//
// When wr_data_enable=1, commands are grouped in 8-command (256-bit) beats.
// After each 8-command beat, N additional 256-bit data beats follow, where
// N = number of WR/WRA commands in that group (max 2: at most 1 per 4-cmd
// sub-group).  A splitter FSM routes command beats to the command FIFO and
// data beats to a separate write-data FIFO.
//
// When the command decoder encounters a 4-command group containing a
// WR/WRA, it dequeues 256 bits from the write-data FIFO and replicates
// them to both pseudo-channel halves of st_ddr_wdata[511:0].
//
// When wr_data_enable=0, behaviour is identical to the original design:
// WR/WRA commands use a fixed DUMMY_WDATA pattern (0xDEADBEEF).
//
// Command FIFO: 256-bit write / 128-bit read (XPM, FWFT).
// Write-data FIFO: 256-bit write / 256-bit read (XPM, FWFT).
//
// Each 32-bit command word is encoded as:
//   [31:28] CMD_TYPE  (4 bits) — project.vh encoding
//   [27:14] ROW       (14 bits)
//   [13:9]  COL       (5 bits)
//   [8:7]   BG        (2 bits)
//   [6:5]   BANK      (2 bits)
//   [4]     PC        (1 bit)  — pseudo-channel (maps to ddr_rank)
//   [3:0]   CH        (4 bits) — HBM channel ID
//
// End marker: any 32-bit slot == 32'hFFFF_FFFF terminates the stream.
// A 64-bit wall-clock cycle counter measures time from first command
// consumed to the end marker.

module stream_test (
  input                                   clk,
  input                                   rst,

  // Control
  input                                   stream_active,

  // AXI-Stream H2C input (fab_clk domain, 256-bit)
  input  [`XDMA_AXI_DATA_WIDTH-1:0]      h2c_tdata,
  input                                   h2c_tvalid,
  output                                  h2c_tready,

  // HBM command outputs (active for one cycle per pop, 4 slots)
  output reg [3:0]                        st_ddr_act,
  output reg [3:0]                        st_ddr_pre,
  output reg [3:0]                        st_ddr_read,
  output reg [3:0]                        st_ddr_write,
  output reg [3:0]                        st_ddr_ref,
  output reg [3:0]                        st_ddr_nop,
  output reg [3:0]                        st_ddr_ap,
  output reg [3:0]                        st_ddr_pall,
  output reg [3:0]                        st_ddr_rank,
  output reg [3:0]                        st_hbm_sel_ch,
  output reg [4*`HBM_CH_WIDTH-1:0]        st_hbm_ch,
  output reg [4*`BG_WIDTH-1:0]            st_ddr_bg,
  output reg [4*`BANK_WIDTH-1:0]          st_ddr_bank,
  output reg [4*`COL_WIDTH-1:0]           st_ddr_col,
  output reg [4*`ROW_WIDTH-1:0]           st_ddr_row,
  output reg [511:0]                      st_ddr_wdata,
  output reg                              cmd_valid,

  // Result
  output reg                              stream_done,
  output reg [63:0]                       stream_cycle_count
);

  // Dummy write data pattern (backward-compat when wr_data_enable=0)
  localparam [511:0] DUMMY_WDATA = {16{32'hDEAD_BEEF}};

  // ---------------------------------------------------------------
  // AXI-to-FIFO splitter
  //
  //  SPLIT_HEADER → SPLIT_CMD ↔ SPLIT_DATA
  //
  //  HEADER : first AXI beat — extract wr_data_enable, push to cmd FIFO
  //  CMD    : 8-command beat  — count writes, push to cmd FIFO
  //  DATA   : write-data beat — push to wdata FIFO (one per write found)
  // ---------------------------------------------------------------

  // Scan incoming AXI beat for WR/WRA commands (only meaningful in CMD state)
  wire [3:0] h2c_cmd_type [0:7];
  wire       h2c_is_wr    [0:7];
  genvar g;
  generate
    for (g = 0; g < 8; g = g + 1) begin : gen_h2c_scan
      assign h2c_cmd_type[g] = h2c_tdata[g*32 + 28 +: 4];
      assign h2c_is_wr[g]    = (h2c_cmd_type[g] == `WR) |
                                (h2c_cmd_type[g] == `WRA);
    end
  endgenerate

  wire [3:0] h2c_wr_count = h2c_is_wr[0] + h2c_is_wr[1] + h2c_is_wr[2]
                           + h2c_is_wr[3] + h2c_is_wr[4] + h2c_is_wr[5]
                           + h2c_is_wr[6] + h2c_is_wr[7];

  // Splitter states
  localparam [1:0] SPLIT_HEADER = 2'd0,
                    SPLIT_CMD    = 2'd1,
                    SPLIT_DATA   = 2'd2;

  reg  [1:0] split_state;
  reg  [3:0] data_beats_remaining;
  reg        wr_data_enabled;

  // FIFO full signals (declared here, driven by FIFO instances below)
  wire cmd_fifo_full;
  wire wdata_fifo_full;

  // AXI handshake
  wire h2c_tready_int = stream_active &
       ((split_state == SPLIT_DATA) ? ~wdata_fifo_full : ~cmd_fifo_full);
  assign h2c_tready = h2c_tready_int;
  wire h2c_accepted = h2c_tvalid & h2c_tready_int;

  // Routing to FIFOs
  wire cmd_fifo_wr_en   = h2c_accepted & (split_state != SPLIT_DATA);
  wire wdata_fifo_wr_en = h2c_accepted & (split_state == SPLIT_DATA);

  always @(posedge clk) begin
    if (rst | ~stream_active) begin
      split_state          <= SPLIT_HEADER;
      data_beats_remaining <= 4'd0;
      wr_data_enabled      <= 1'b0;
    end
    else if (h2c_accepted) begin
      case (split_state)
        SPLIT_HEADER: begin
          wr_data_enabled <= h2c_tdata[32];   // bit 0 of flags word
          split_state     <= SPLIT_CMD;
        end
        SPLIT_CMD: begin
          if (wr_data_enabled && h2c_wr_count != 0) begin
            split_state          <= SPLIT_DATA;
            data_beats_remaining <= h2c_wr_count;
          end
        end
        SPLIT_DATA: begin
          if (data_beats_remaining == 4'd1)
            split_state <= SPLIT_CMD;
          data_beats_remaining <= data_beats_remaining - 4'd1;
        end
        default: split_state <= SPLIT_HEADER;
      endcase
    end
  end

  // ---------------------------------------------------------------
  // Command FIFO  (FIFO Generator IP, 256-bit write / 128-bit read, FWFT)
  // ---------------------------------------------------------------
  wire                cmd_fifo_rd_en;
  wire [127:0]        fifo_dout;
  wire                cmd_fifo_empty;

  stream_cmd_fifo u_cmd_fifo (
    .clk   (clk),
    .srst  (rst | ~stream_active),
    .din   (h2c_tdata),
    .wr_en (cmd_fifo_wr_en),
    .rd_en (cmd_fifo_rd_en),
    .dout  (fifo_dout),
    .full  (cmd_fifo_full),
    .empty (cmd_fifo_empty)
  );

  // ---------------------------------------------------------------
  // Write-data FIFO  (FIFO Generator IP, 256-bit symmetric, FWFT)
  // ---------------------------------------------------------------
  wire                wdata_fifo_rd_en;
  wire [255:0]        wdata_fifo_out;
  wire                wdata_fifo_empty;

  stream_wdata_fifo u_wdata_fifo (
    .clk   (clk),
    .srst  (rst | ~stream_active),
    .din   (h2c_tdata),
    .wr_en (wdata_fifo_wr_en),
    .rd_en (wdata_fifo_rd_en),
    .dout  (wdata_fifo_out),
    .full  (wdata_fifo_full),
    .empty (wdata_fifo_empty)
  );

  // ---------------------------------------------------------------
  // End-of-stream detection
  // ---------------------------------------------------------------
  wire end_marker_hit = (fifo_dout[ 31:  0] == 32'hFFFF_FFFF) |
                        (fifo_dout[ 63: 32] == 32'hFFFF_FFFF) |
                        (fifo_dout[ 95: 64] == 32'hFFFF_FFFF) |
                        (fifo_dout[127: 96] == 32'hFFFF_FFFF);

  // ---------------------------------------------------------------
  // Extract 4 × 32-bit commands from cmd FIFO output
  // ---------------------------------------------------------------
  wire [31:0] cmd [0:3];
  assign cmd[0] = fifo_dout[ 31:  0];
  assign cmd[1] = fifo_dout[ 63: 32];
  assign cmd[2] = fifo_dout[ 95: 64];
  assign cmd[3] = fifo_dout[127: 96];

  // Detect WR/WRA in the current 4-command group
  wire any_write_in_group = ((cmd[0][31:28] == `WR) | (cmd[0][31:28] == `WRA))
                          | ((cmd[1][31:28] == `WR) | (cmd[1][31:28] == `WRA))
                          | ((cmd[2][31:28] == `WR) | (cmd[2][31:28] == `WRA))
                          | ((cmd[3][31:28] == `WR) | (cmd[3][31:28] == `WRA));

  // ---------------------------------------------------------------
  // Stall & pop control
  // ---------------------------------------------------------------
  wire cmd_fifo_valid = ~cmd_fifo_empty;

  // Stall the command decoder when write data is needed but not yet available
  wire write_data_needed = cmd_fifo_valid & ~done_latched & ~end_marker_hit
                         & wr_data_enabled & any_write_in_group;
  wire stall_for_wdata   = write_data_needed & wdata_fifo_empty;

  // Pop command FIFO
  assign cmd_fifo_rd_en = cmd_fifo_valid & stream_active
                        & ~done_latched & ~stall_for_wdata;

  // Pop write-data FIFO (one entry per 4-cmd group that contains a write)
  assign wdata_fifo_rd_en = cmd_fifo_rd_en & wr_data_enabled
                          & any_write_in_group & ~wdata_fifo_empty;

  // ---------------------------------------------------------------
  // Registered write-data hold
  //
  // The HBM cmd_gen pipeline may sample ddr_wdata one or more cycles
  // AFTER the WR command cycle.  Keep st_ddr_wdata at the last
  // write-data value so the adapter always finds it on the bus.
  // ---------------------------------------------------------------
  reg [511:0] wdata_hold_r;

  always @(posedge clk) begin
    if (rst | ~stream_active)
      wdata_hold_r <= DUMMY_WDATA;
    else if (cmd_fifo_rd_en & any_write_in_group) begin
      if (wr_data_enabled & ~wdata_fifo_empty)
        wdata_hold_r <= {wdata_fifo_out, wdata_fifo_out};
      else
        wdata_hold_r <= DUMMY_WDATA;
    end
  end

  // ---------------------------------------------------------------
  // State machine & cycle counter
  // ---------------------------------------------------------------
  reg        consuming_r;
  reg        done_latched;
  reg [63:0] cycle_cnt_r;

  always @(posedge clk) begin
    if (rst | ~stream_active) begin
      consuming_r        <= 1'b0;
      done_latched       <= 1'b0;
      stream_done        <= 1'b0;
      cycle_cnt_r        <= 64'd0;
      stream_cycle_count <= 64'd0;
    end
    else begin
      stream_done <= 1'b0;

      if (cmd_fifo_valid & ~done_latched & ~stall_for_wdata) begin
        if (end_marker_hit) begin
          stream_done        <= 1'b1;
          done_latched       <= 1'b1;
          stream_cycle_count <= cycle_cnt_r + 64'd1;
          consuming_r        <= 1'b0;
        end
        else if (~consuming_r)
          consuming_r <= 1'b1;
      end

      if (consuming_r)
        cycle_cnt_r <= cycle_cnt_r + 64'd1;
    end
  end

  // ---------------------------------------------------------------
  // Combinational command decode
  // ---------------------------------------------------------------
  integer i;
  reg [3:0]               cmd_type;
  reg [`ROW_ADDR_WIDTH-1:0] cmd_row;
  reg [`COL_ADDR_WIDTH-1:0] cmd_col;
  reg [`BG_WIDTH-1:0]      cmd_bg;
  reg [`BANK_WIDTH-1:0]    cmd_bank;
  reg                       cmd_pc;
  reg [`HBM_CH_WIDTH-1:0]  cmd_ch;

  always @(*) begin
    // Defaults: all NOPs, zero addresses
    st_ddr_act    = 4'b0;
    st_ddr_pre    = 4'b0;
    st_ddr_read   = 4'b0;
    st_ddr_write  = 4'b0;
    st_ddr_ref    = 4'b0;
    st_ddr_nop    = 4'b1111;
    st_ddr_ap     = 4'b0;
    st_ddr_pall   = 4'b0;
    st_ddr_rank   = 4'b0;
    st_hbm_sel_ch = 4'b0;
    st_hbm_ch     = {(4*`HBM_CH_WIDTH){1'b0}};
    st_ddr_bg     = {(4*`BG_WIDTH){1'b0}};
    st_ddr_bank   = {(4*`BANK_WIDTH){1'b0}};
    st_ddr_col    = {(4*`COL_WIDTH){1'b0}};
    st_ddr_row    = {(4*`ROW_WIDTH){1'b0}};
    st_ddr_wdata  = wdata_hold_r;   // persistent: adapter may sample after WR cycle
    cmd_valid     = 1'b0;

    if (cmd_fifo_valid & ~done_latched & ~end_marker_hit & ~stall_for_wdata) begin
      cmd_valid = 1'b1;

      // Write data: combinational override on WR cycle (same-cycle capture)
      // wdata_hold_r already stores the value for next-cycle capture.
      if (any_write_in_group) begin
        if (wr_data_enabled & ~wdata_fifo_empty)
          st_ddr_wdata = {wdata_fifo_out, wdata_fifo_out};
        else
          st_ddr_wdata = DUMMY_WDATA;
      end

      for (i = 0; i < 4; i = i + 1) begin
        cmd_type = cmd[i][31:28];
        cmd_row  = cmd[i][27:14];
        cmd_col  = cmd[i][13:9];
        cmd_bg   = cmd[i][8:7];
        cmd_bank = cmd[i][6:5];
        cmd_pc   = cmd[i][4];
        cmd_ch   = cmd[i][3:0];

        // Address fields (zero-padded to full bus width)
        st_ddr_row [`ROW_WIDTH*i  +: `ROW_WIDTH]  = {{(`ROW_WIDTH-`ROW_ADDR_WIDTH){1'b0}}, cmd_row};
        st_ddr_col [`COL_WIDTH*i  +: `COL_WIDTH]   = {{(`COL_WIDTH-`COL_ADDR_WIDTH){1'b0}}, cmd_col};
        st_ddr_bg  [`BG_WIDTH*i   +: `BG_WIDTH]    = cmd_bg;
        st_ddr_bank[`BANK_WIDTH*i +: `BANK_WIDTH]  = cmd_bank;
        st_ddr_rank[i]                              = cmd_pc;
        st_hbm_ch  [`HBM_CH_WIDTH*i +: `HBM_CH_WIDTH] = cmd_ch;
        st_hbm_sel_ch[i]                            = 1'b1;

        // Decode command type
        st_ddr_nop[i] = 1'b0;
        case (cmd_type)
          `ACTT:    st_ddr_act[i]   = 1'b1;
          `PREE:    st_ddr_pre[i]   = 1'b1;
          `PREA:    begin st_ddr_pre[i] = 1'b1; st_ddr_pall[i] = 1'b1; end
          `RD:      st_ddr_read[i]  = 1'b1;
          `RDA:     begin st_ddr_read[i] = 1'b1; st_ddr_ap[i] = 1'b1; end
          `WR:      st_ddr_write[i] = 1'b1;
          `WRA:     begin st_ddr_write[i] = 1'b1; st_ddr_ap[i] = 1'b1; end
          `REFF:    st_ddr_ref[i]   = 1'b1;
          default:  st_ddr_nop[i]   = 1'b1;
        endcase
      end
    end
  end

endmodule
