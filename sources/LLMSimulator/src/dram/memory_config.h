#pragma once

namespace llm_system {
class MemoryConfig {
 public:
 MemoryConfig& operator=(const MemoryConfig& rhs) = default;

  // 16GB per cube, total 80GB HBM configuration & 1 cube, 16Gb density, 2 channel, LPDDR5
  MemoryConfig(int num_cube = 5, int num_logic_cube = 5, int num_channel = 32, int num_rank = 2,
               int num_bankgroup = 4, int num_bank = 4, int num_row = 16384,
               int num_col = 32, int granul = 32)
        : num_cube(num_cube),
          num_logic_cube(num_logic_cube),
          num_channel(num_channel),
          num_rank(num_rank),
          num_bankgroup(num_bankgroup),
          num_bank(num_bank),
          num_row(num_row),
          num_col(num_col),
          granul(granul){};

    int num_cube;
    int num_logic_cube;
    
    int num_channel;
    int num_rank;
    int num_bankgroup;
    int num_bank;
    int num_row;
    int num_col;
    int granul;
  };

  static MemoryConfig hbm3_80GB = MemoryConfig(
    5,      // num_cube 
    5,      // num_logic_cube
    32,     // num_channel
    2,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  static MemoryConfig hbm3e_192GB = MemoryConfig(
    8,      // num_cube
    8,      // num_logic_cube
    32,     // num_channel
    3,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  // A100 80GB HBM2e. Same 80GB geometry as hbm3_80GB; paired with the HBM2E
  // Ramulator model (dram_config_HBM2e_80GB.yaml). num_rank/row/col stay within
  // the HBM2E_A100_80GB org (rank 2, row 1<<15, col 1<<5).
  static MemoryConfig hbm2e_80GB = MemoryConfig(
    5,      // num_cube
    5,      // num_logic_cube
    32,     // num_channel
    2,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  // A100 40GB HBM2: 8Gb 8-Hi dies (8 GB/stack) x 5 stacks, pseudochannel mode.
  //   bank = SID, BA[3:0]  -> rank(1, SID dropped) x bankgroup(4) x bank(4, =BA[3:0])
  //   row  = RA[13:0]      -> 16384
  //   col  = CA[5:1]       -> 32
  //   256-bit prefetch (=32B granule); 32 col x 256b = 1KB page
  //   8 HBM2 channels x 2 pseudochannels per stack -> num_channel = 16
  //   capacity = 5*16*1*4*4*16384*32*32 B = 20 GiB  (1 rank)
  static MemoryConfig hbm2_40GB = MemoryConfig(
    5,      // num_cube       (5 stacks)
    5,      // num_logic_cube
    16,     // num_channel    (8 channels x 2 pseudochannels)
    1,      // num_rank       (single rank, SID dropped)
    4,      // num_bankgroup  (BA[3:2])
    4,      // num_bank       (BA[1:0])
    16384,  // num_row        (RA[13:0])
    32,     // num_col        (CA[5:1])
    32     // granul          (256-bit prefetch = 32B)
  );

}  // namespace llm_system