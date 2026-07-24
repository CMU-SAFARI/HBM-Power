#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <list>
#include <set>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <csignal>

// A 4 GB per stack device has 4 Gb per channel, and each channel has
// two 2 Gb or 256 MB pseudo channels.

// THIS LOOKS LIKE THE CORRECT ORGANIZATION:
// 32 byte per column x 32 (cols) x 16K (rows) x 16 (banks) x 2 PC x 8 channels

using namespace std;

#define USE_SMC_FILE 0

#define PC0 0
#define PC1 1

#define NUM_CH 8
#define NUM_PC 2
#define NUM_BANKS 16
#define NUM_ROWS 1024*16
#define NUM_COLS 32
#define BYTES_PER_READ 32
#define ROW_SIZE (NUM_COLS*BYTES_PER_READ)

// Stride register ids are fixed and should not be changed
// CASR should always be reg 0
// BASR should always be reg 1
// RASR should always be reg 2
#define CASR 0
#define BASR 1
#define RASR 2

// #define bg0 0
// #define bg1 1

#define BA0 3
#define BA1 4
#define BA2 5
#define BA3 6
#define BA4 7
#define BA5 8
#define BA6 9
#define BA7 10

#define RA5555 11
#define RA2AAA 12

#define RA0000 11
#define RA3FFF 12

#define CA01010 13
#define CA10101 14

#define CA00000 13
#define CA11111 14

#define CA2_REG 5  // 3rd column register (reuses BA2, safe in 2-bank modes)

#define PATTERN_REG 15
#define LOOP_REG 15
#define UB_REG 12

#define NUM_REPS 1024 * 1024
#define INNER_REPS 1024 * 16
// #define NUM_REPS 1
// #define INNER_REPS 1
#define BUFFER_SIZE (16 * NUM_PC * BYTES_PER_READ)

int multi_channel, pattern, access_pattern;

#define MULTI_CHANNEL 1
#define PATTERN pattern
// 0 for IDD4R
// 1 for IDD4W
#define ACCESS_PATTERN access_pattern
int garbage_reads_enabled = 1;
#define GARBAGE_READS garbage_reads_enabled

uint32_t wr_pattern;
uint32_t custom_wr_pattern = 0x00000000; // Custom pattern for PATTERN 4/5
bool shift_pattern = true; // Whether to shift wr_pattern in programming functions
bool use_multi_word_pattern = false; // Whether to use multi_word_pattern[] instead of wr_pattern
uint32_t multi_word_pattern[16] = {0}; // Per-word pattern for PATTERN 6

uint32_t pattern_32b[8] = {0}; // 32-byte pattern for PATTERN 7 (8 x uint32_t)
uint32_t pattern_32b_B[8] = {0}; // 2nd data pattern for 3-col mode
uint32_t pattern_32b_C[8] = {0}; // 3rd data pattern for 3-col mode
uint8_t bitflip_mask = 0x00;   // 8-bit mask for bitflip test (bit i = flip between col i and col i+1)
bool invert_col1 = false;      // Whether col1 addresses get inverted data pattern (for idd4r_custom_data)

uint32_t zeros = 0x00000000; //  PATTERN 0
uint32_t ones  = 0xffffffff; //  PATTERN 1
uint32_t idd4r = 0x0055ffaa; //  PATTERN 2
uint32_t all_as = 0xaaaaaaaa; // PATTERN 3

std::vector<int> channels_to_broadcast;
Program program;

void load_pattern_to_program() {
	if(use_multi_word_pattern) {
		for(uint16_t i = 0; i < 16; i++) {
			program.add_inst(SMC_LI(multi_word_pattern[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
	} else {
		for(uint16_t i = 0; i < 16; i++) {
			program.add_inst(SMC_LI(wr_pattern, PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
			if(shift_pattern) wr_pattern = (wr_pattern >> 8) | (wr_pattern << 24);
		}
	}
}

// Global platform pointer for signal handler
static SoftMCPlatform *g_platform = nullptr;

void sigint_handler(int sig)
{
	std::cout << "\n[SIGINT] Caught Ctrl+C, cleaning up..." << std::endl;
	if (g_platform) {
		g_platform->stopMetricsThread();
		g_platform->reset_fpga();
		std::cout << "[SIGINT] FPGA reset complete." << std::endl;
	}
	_exit(0);
}

// Configurable parameters for structural variation tests
int bank_offsets[4] = {0, 1, 2, 3};
uint16_t row_addr0 = 0x5555;
uint16_t row_addr1 = 0x2aaa;
uint16_t col_addr0 = 0b01010;
uint16_t col_addr1 = 0b10101;
uint16_t col_addr2 = 0b11111; // 3rd column address for 3-col mode
bool use_full_idd4r = false;
bool use_bank_variation = false;
int fixed_duration_s = 0; // 0 = use stabilization (default), >0 = fixed-duration mode
int freq_mhz = 0; // 0 = no frequency tag in CSV, >0 = append _freqXXXMHz

// Whole-DRAM pre-fill: write a chosen data pattern to EVERY (bank,row,col) of
// both pseudo-channels before the measurement, so resident-data-dependent tests
// (IDD0/2/3N/5) see a controlled background. 0 = off, 1 = zeros, 2 = per-row
// random (each row a different, seed-derived pattern, reproducible across FPGAs).
int fill_whole_mode = 0;
uint32_t fill_seed = 0xBADC0FFE;

// buffer allocated for reading data from the board
uint8_t buf[BUFFER_SIZE]; // * 2 because of PCs

/**
 * @return an instruction formed by NOPs
 */
Inst all_nops()
{
	return  __pack_mininsts(SMC_NOP(), SMC_NOP(), SMC_NOP(), SMC_NOP());
}

void program_idd0(int ch, int pc, int bg0, int bg1) {
	// Assuming 600MHz HBM frequency => 1.67ns per cycle
	// According to JEDEC standard:
	// tRAS=33ns = 20 cycles
	// tRC=48ns = 29 cycles
	// tRP=15ns = 9 cycles

	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP()); // Select target channel

	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

    program.add_inst(SMC_LI(row_addr0, RA5555));
    program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));

    for(int i = 0; i < 4; i++) {
        program.add_inst(SMC_LI(i, BA0 + i));
    }

	program.add_inst(SMC_PRE(BA0, 0, 1, pc)); // First precharge all banks
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	// Wait for tRP -- I think we already wait 7 cycles until we get here, so 4 cycles should be enough (?)
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	//  Now issue ACT commands to all banks
	
	for(int j = 0; j < 2; j++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	// Wait for tRCD -- Probably unnecessary as there are already enough cycles between the ACT and WRITE commands thanks to the loop
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Now issue WRITE commands to all banks
	for(int j = 0; j < 2; j++) {
		for(int i = 0; i < 4; i++) {
			for(int k = 0; k < 30; k++) {
				program.add_inst(SMC_WRITE(BA0 + i, 0, CA00000, 1, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			}

			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 1, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			for(int k = 0; k < 30; k++) {
				program.add_inst(SMC_WRITE(BA0 + i, 0, CA00000, 1, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			}
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 1, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

    program.add_label("IDD0_LOOP");

    // ==========================
    // ACT PHASE
    // ==========================
	// 2 cycles of ACT followed by tRAS-1 = 19 cycles of NOPs

	for(int j = 0; j < INNER_REPS / 64; j++) {
		vector <int> act_order = {0, 5, 2, 7, 1, 6, 3, 4}; // Interleave ACTs from the two BGs to maximize current

		for(int i = 0; i < 16; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555; // Alternate between two rows
			program.add_inst(SMC_ACT(BA0 + act_order[i % 8], 0, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			// Then a PRE command followed by tRP-2 = 8 cycles of NOPs
			// Add + 8 to the bank to have BA3=1 in the next iteration
			program.add_inst(SMC_NOP(), SMC_PRE(BA0 + act_order[i], 1, 0, pc), SMC_NOP(), SMC_NOP());
			program.add_inst(SMC_NOP(), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_mininst(SMC_NOP(), 0);
			program.pack_minprogram();
		}
	}

    program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD0_LOOP");
}

void program_idd4r(int ch, int pc, int bg0, int bg1) {
	// First load the pattern to be written

	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP()); // Select target channel

	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i)); // Initialize BA registers
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// First precharge all banks
	// for(int i = 0; i < 16; i += 8) {
	// 	program.add_inst(SMC_PRE(BA0 + i, 0, 0, pc), SMC_PRE(BA1 + i, 0, 0, pc), SMC_PRE(BA2 + i, 0, 0, pc), SMC_PRE(BA3 + i, 0, 0, pc));
	// 	program.add_inst(SMC_PRE(BA4 + i, 0, 0, pc), SMC_PRE(BA5 + i, 0, 0, pc), SMC_PRE(BA6 + i, 0, 0, pc), SMC_PRE(BA7 + i, 0, 0, pc));
	// }
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	// Wait for tRP -- I think we already wait 7 cycles until we get here, so 4 cycles should be enough (?)
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	//  Now issue ACT commands to all banks
	
	// for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	// }

	// Wait for tRCD -- Probably unnecessary as there are already enough cycles between the ACT and WRITE commands thanks to the loop
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Now issue WRITE commands to all banks
	// for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	// }

	// Now issue READ commands according to the JEDEC IDD4R pattern (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
	program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA5, 0, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA7, 0, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA6, 0, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA3, 0, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

void program_idd4w(int ch, int pc, int bg0, int bg1) {
	// First load the pattern to be written

	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP()); // Select target channel

	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i)); // Initialize BA registers
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// First precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	// Wait for tRP
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	//  Now issue ACT commands to all banks

	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	// Wait for tRCD
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Initial WRITE to all banks
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	// Now issue WRITE commands according to the JEDEC IDD4W pattern (infinite loop)
	program.add_label("WRITE_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
	program.add_inst(SMC_WRITE(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_WRITE(BA5, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_WRITE(BA7, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_WRITE(BA6, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_WRITE(BA3, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_WRITE(BA4, 1, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "WRITE_PHASE");
}

void program_idd4r_full(int ch, int pc, int bg0, int bg1) {
	// First load the pattern to be written

	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP()); // Select target channel

	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	// for(uint16_t i = 0 ; i < 4 ; i++) {
	// 	for(int j = 0; j < 4; j++) {
	// 		program.add_inst(SMC_LI(wr_pattern, PATTERN_REG)); 
	// 		program.add_inst(SMC_LDWD(PATTERN_REG, 4 * i + j));
	// 	}
	// 	wr_pattern = (wr_pattern >> 8) | (wr_pattern << 24);
	// }

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i)); // Initialize BA registers
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// First precharge all banks
	// for(int i = 0; i < 16; i += 8) {
	// 	program.add_inst(SMC_PRE(BA0 + i, 0, 0, pc), SMC_PRE(BA1 + i, 0, 0, pc), SMC_PRE(BA2 + i, 0, 0, pc), SMC_PRE(BA3 + i, 0, 0, pc));
	// 	program.add_inst(SMC_PRE(BA4 + i, 0, 0, pc), SMC_PRE(BA5 + i, 0, 0, pc), SMC_PRE(BA6 + i, 0, 0, pc), SMC_PRE(BA7 + i, 0, 0, pc));
	// }
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	// Wait for tRP -- I think we already wait 7 cycles until we get here, so 4 cycles should be enough (?)
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	//  Now issue ACT commands to all banks
	
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
		}
	}

	// Wait for tRCD -- Probably unnecessary as there are already enough cycles between the ACT and WRITE commands thanks to the loop
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Now issue WRITE commands to all banks
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Now issue READ commands according to the JEDEC IDD4R pattern (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
	program.add_inst(SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA2, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA1, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA3, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// Same as program_idd4r_full but covers all 32 banks (SID bit enabled in HW).
// With BASR=8, 4 outer iterations × 8 BA registers wrap through both SIDs.
void program_idd4r_full_32bank(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	for(int j = 0; j < 32; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
			program.add_inst(all_nops());
		}
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	for(int j = 0; j < 32; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}

		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
	program.add_inst(SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA2, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA1, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0), SMC_NOP());
	program.add_inst(SMC_READ(BA3, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// Bank variation: hammer exactly 4 banks, one per bank group.
// Given offsets a, b, c, d (bank_offsets[0..3]), access pattern alternates:
//   BG0 bank a (0+a), BG2 bank c (8+c), BG1 bank b (4+b), BG3 bank d (12+d)
void program_idd4r_bank_variation(int ch, int pc) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Bank variation mode: banks [%d, %d, %d, %d] (BG0+%d, BG1+%d, BG2+%d, BG3+%d)\n",
		0 + bank_offsets[0], 4 + bank_offsets[1], 8 + bank_offsets[2], 12 + bank_offsets[3],
		bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3]);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	// BA0 = BG0 bank a = 0*4 + a
	// BA1 = BG1 bank b = 1*4 + b
	// BA2 = BG2 bank c = 2*4 + c
	// BA3 = BG3 bank d = 3*4 + d
	program.add_inst(SMC_LI(0 * 4 + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(1 * 4 + bank_offsets[1], BA1));
	program.add_inst(SMC_LI(2 * 4 + bank_offsets[2], BA2));
	program.add_inst(SMC_LI(3 * 4 + bank_offsets[3], BA3));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses across the 4 banks
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA3, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to all 4 banks
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA3, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// READ phase: alternate BG0(a), BG2(c), BG1(b), BG3(d) (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS * 2; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA2, 0, CA01010, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA3, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 4 banks in 2 bank groups (2 banks per BG).
// BA0 = bg0 + bank_offsets[0], BA1 = bg0 + bank_offsets[1]
// BA2 = bg1 + bank_offsets[0], BA3 = bg1 + bank_offsets[1]
void program_idd4r_4bank_2bg(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD4R 4-bank 2-BG mode: banks [%d, %d, %d, %d] (BG%d+%d, BG%d+%d, BG%d+%d, BG%d+%d)\n",
		4*bg0 + bank_offsets[0], 4*bg0 + bank_offsets[1],
		4*bg1 + bank_offsets[0], 4*bg1 + bank_offsets[1],
		bg0, bank_offsets[0], bg0, bank_offsets[1],
		bg1, bank_offsets[0], bg1, bank_offsets[1]);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[1], BA1));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[0], BA2));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[1], BA3));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses across the 4 banks
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA2, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA3, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to all 4 banks
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA3, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// READ phase: pair reads across BGs (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS * 2; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA2, 0, CA01010, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA3, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 2 banks, 1 per bank group.
// BA0 = bg0 + bank_offsets[0], BA1 = bg1 + bank_offsets[0]
void program_idd4r_2bank(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD4R 2-bank mode: banks [%d, %d] (BG%d+%d, BG%d+%d)\n",
		4*bg0 + bank_offsets[0], 4*bg1 + bank_offsets[1],
		bg0, bank_offsets[0], bg1, bank_offsets[1]);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[1], BA1));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to both banks
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// READ phase: pair reads across BGs (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS * 4; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA1, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 3 banks, 1 per bank group across 3 BGs (one BG excluded).
// bg0 = index of the MISSING bank group (0-3).
// The 3 active BGs are the remaining ones.
// BA0, BA1, BA2 = one bank from each active BG (same offset = bank_offsets[0]).
// Read pattern: 2 reads paired + 1 read solo per 2 cycles.
void program_idd4r_3bank_3bg(int ch, int pc, int missing_bg) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// Derive the 3 active BGs
	int active_bgs[3];
	int idx = 0;
	for(int bg = 0; bg < 4; bg++) {
		if(bg != missing_bg) active_bgs[idx++] = bg;
	}

	printf("IDD4R 3-bank 3-BG mode: banks [%d, %d, %d] (BG%d+%d, BG%d+%d, BG%d+%d), missing BG%d\n",
		4*active_bgs[0] + bank_offsets[0],
		4*active_bgs[1] + bank_offsets[0],
		4*active_bgs[2] + bank_offsets[0],
		active_bgs[0], bank_offsets[0],
		active_bgs[1], bank_offsets[0],
		active_bgs[2], bank_offsets[0],
		missing_bg);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_LI(4 * active_bgs[0] + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(4 * active_bgs[1] + bank_offsets[0], BA1));
	program.add_inst(SMC_LI(4 * active_bgs[2] + bank_offsets[0], BA2));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses across the 3 banks
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA2, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to all 3 banks, both columns (each bank is read at both CA01010 and CA10101)
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA0, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// READ phase: 2 reads per cycle, 3-bank rotation (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA1, 0, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA0, 0, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA2, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 8 banks, 2 per bank group across all 4 BGs.
// BA0 = bg0+off[0], BA1 = bg0+off[1], BA2 = bg1+off[0], BA3 = bg1+off[1]
// With ibar=1 and BASR=8, BA4..BA7 = BA0..BA3 + 8, covering BG2 and BG3.
void program_idd4r_8bank_4bg(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD4R 8-bank 4-BG mode: BG%d+[%d,%d], BG%d+[%d,%d], BG%d+[%d,%d], BG%d+[%d,%d]\n",
		bg0, bank_offsets[0], bank_offsets[1],
		bg1, bank_offsets[0], bank_offsets[1],
		bg0+2, bank_offsets[0], bank_offsets[1],
		bg1+2, bank_offsets[0], bank_offsets[1]);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	// BA0..BA3 = 2 banks in BG0 + 2 banks in BG1
	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[1], BA1));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[0], BA2));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[1], BA3));
	// BA4..BA7 auto-computed via ibar=1: +8 → 2 banks in BG2 + 2 banks in BG3

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 8 banks (2 passes via ibar=1)
	for(int j = 0; j < 16; j += 8) {
		program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_ACT(BA1, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_ACT(BA3, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE all 8 banks (2 passes via ibar=1)
	for(int j = 0; j < 16; j += 8) {
		program.add_inst(SMC_WRITE(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_WRITE(BA1, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_WRITE(BA2, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_WRITE(BA3, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// READ phase: pair reads across BGs with ibar=1 (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
		program.add_inst(SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA3, 1, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 1, CA10101, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA2, 1, CA01010, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 2 banks, 1 per bank group, sparse reads (1 RD per 4 cycles per bank).
// Instead of pairing reads (RD BA0, NOP, RD BA1, NOP) we space them:
//   (RD BA0, NOP, NOP, NOP), (RD BA1, NOP, NOP, NOP)
// so each bank is read once every 8 slots = 4 cycles.
void program_idd4r_2bank_sparse(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD4R 2-bank sparse mode: banks [%d, %d] (BG%d+%d, BG%d+%d)\n",
		4*bg0 + bank_offsets[0], 4*bg1 + bank_offsets[0],
		bg0, bank_offsets[0], bg1, bank_offsets[0]);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0], BA0));
	program.add_inst(SMC_LI(4 * bg1 + bank_offsets[0], BA1));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to both banks
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// READ phase: one read per instruction, NOPs fill the rest (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS * 2; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// IDD4R with 4 banks in a single bank group.
// BA0 = bg0*4+0, BA1 = bg0*4+1, BA2 = bg0*4+2, BA3 = bg0*4+3
// Single-BG can only issue 1 read per cycle (no cross-BG pairing).
void program_idd4r_4bank_1bg(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD4R 4-bank 1-BG mode: banks [%d, %d, %d, %d] in BG%d\n",
		4*bg0 + 0, 4*bg0 + 1, 4*bg0 + 2, 4*bg0 + 3, bg0);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	program.add_inst(SMC_LI(4 * bg0 + 0, BA0));
	program.add_inst(SMC_LI(4 * bg0 + 1, BA1));
	program.add_inst(SMC_LI(4 * bg0 + 2, BA2));
	program.add_inst(SMC_LI(4 * bg0 + 3, BA3));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT: alternate row addresses across the 4 banks
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA1, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA2, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA3, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE to all 4 banks (column must match what READ will access)
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA2, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA3, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// READ phase: 2 reads per cycle, cycle through banks (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
		program.add_inst(SMC_READ(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_READ(BA2, 0, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_READ(BA3, 0, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

void program_idd2(int ch, int pc) {
	// IDD2: standby current — precharge all banks, then infinite NOP loop
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(SMC_LI(0, BA0));
	program.add_inst(SMC_LI(8, BASR));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite NOP loop (IDD2 idle — all banks precharged)
	program.add_label("IDD2_IDLE");
	for(int j = 0; j < 64; j++) {
		program.add_inst(all_nops());
	}
	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD2_IDLE");
}

void program_idd3n1(int ch, int pc) {
	// IDD3N1: active standby current — activate one row in one bank, then infinite NOP loop
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(SMC_LI(0, BA0));
	program.add_inst(SMC_LI(8, BASR));
	program.add_inst(SMC_LI(row_addr0, RA0000));

	// Precharge all banks first
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Activate one row in bank 0 only
	program.add_inst(SMC_LI(0, BA0));
	program.add_inst(SMC_ACT(BA0, 0, RA0000, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite NOP loop (IDD3N1 active standby — one row open in one bank)
	program.add_label("IDD3N1_IDLE");
	for(int j = 0; j < 64; j++) {
		program.add_inst(all_nops());
	}
	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD3N1_IDLE");
}

void program_idd3n16(int ch, int pc) {
	// IDD3N16: active standby current — activate one row per bank (all 16 banks), then infinite NOP loop
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(SMC_LI(0, BA0));
	program.add_inst(SMC_LI(8, BASR));
	program.add_inst(SMC_LI(row_addr0, RA0000));

	// Precharge all banks first
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Activate one row in each of the 16 banks
	for(int i = 0; i < 16; i++) {
		program.add_inst(SMC_LI(i, BA0));
		program.add_inst(SMC_ACT(BA0, 0, RA0000, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		program.add_inst(all_nops());
		program.add_inst(all_nops());
	}

	// Infinite NOP loop (IDD3N16 active standby — one row open per bank)
	program.add_label("IDD3N16_IDLE");
	for(int j = 0; j < 64; j++) {
		program.add_inst(all_nops());
	}
	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD3N16_IDLE");
}

void program_idd3n16_cooldown(Program &cooldown_prog, int ch, int pc) {
	// IDD3N16 cooldown: activate one row per bank, then infinite NOP loop
	cooldown_prog.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	cooldown_prog.add_inst(SMC_LI(0, BA0));
	cooldown_prog.add_inst(SMC_LI(8, BASR));
	cooldown_prog.add_inst(SMC_LI(row_addr0, RA0000));

	cooldown_prog.add_inst(SMC_PRE(BA0, 0, 1, pc));
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());

	for(int i = 0; i < 16; i++) {
		cooldown_prog.add_inst(SMC_LI(i, BA0));
		cooldown_prog.add_inst(SMC_ACT(BA0, 0, RA0000, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		cooldown_prog.add_inst(all_nops());
		cooldown_prog.add_inst(all_nops());
	}

	cooldown_prog.add_label("IDD3N16_IDLE");
	for(int j = 0; j < 64; j++) {
		cooldown_prog.add_inst(all_nops());
	}
	cooldown_prog.add_branch(cooldown_prog.BR_TYPE::JUMP, 0, 0, "IDD3N16_IDLE");
}

void program_idd2_cooldown(Program &cooldown_prog, int ch, int pc) {
	// IDD2: standby current -- precharge all banks, then infinite NOP loop
	cooldown_prog.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	cooldown_prog.add_inst(SMC_LI(0, BA0));
	cooldown_prog.add_inst(SMC_LI(8, BASR));

	// Precharge all banks
	cooldown_prog.add_inst(SMC_PRE(BA0, 0, 1, pc));
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());
	cooldown_prog.add_inst(all_nops());

	// Infinite NOP loop (IDD2 idle)
	cooldown_prog.add_label("IDD2_IDLE");
	for(int j = 0; j < 64; j++) {
		cooldown_prog.add_inst(all_nops());
	}
	cooldown_prog.add_branch(cooldown_prog.BR_TYPE::JUMP, 0, 0, "IDD2_IDLE");
}

// ---------------------------------------------------------------------------
// Whole-DRAM pre-fill
// ---------------------------------------------------------------------------
// Write a data pattern to EVERY (bank, row, column) of one pseudo-channel, for
// the currently broadcast channels. Modeled on the whole-memory write loop in
// apps/RetentionTest/retention.cpp: a register-driven bank x row loop with the
// 32 column writes unrolled, using the stride registers (CASR/RASR) so CAR/RAR
// auto-increment. Ends with SMC_END so each fill execution runs to completion
// before the caller loads the next program.
//
//   fill_mode 1 = zeros  : load 0 into the wide write register once.
//   fill_mode 2 = random : derive a different 64-byte pattern per row from
//                          (seed-state XOR row), rotating it across the 16 wide
//                          words. Pure arithmetic on the row index + host seed,
//                          so the row->pattern mapping is identical on every FPGA.
//
// Registers (0/1/2 are the special stride regs; 3-15 are general purpose):
//   CASR=0 (col stride), BASR=1, RASR=2 (row stride)
//   r3=bank  r4=col  r5=row(==ACT's RAR)  r6=row_limit  r7=bank_limit
//   r9=random state  r10=per-row mix  r15=scratch (zeros load)
void build_fill_program(Program &p, int ch, int pc, int fill_mode, uint32_t seed) {
	const int FR_BANK    = 3;
	const int FR_COL     = 4;
	const int FR_ROW     = 5;   // also the RAR used by SMC_ACT (irar=1 increments it)
	const int FR_ROWLIM  = 6;
	const int FR_BANKLIM = 7;
	const int FR_STATE   = 9;
	const int FR_MIX     = 10;
	const int FR_DATA    = PATTERN_REG; // 15

	p.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// Stride registers: column +1 per write (icar), row +1 per ACT (irar).
	p.add_inst(SMC_LI(1, CASR));
	p.add_inst(SMC_LI(1, BASR));
	p.add_inst(SMC_LI(1, RASR));

	p.add_inst(SMC_LI(NUM_ROWS, FR_ROWLIM));
	p.add_inst(SMC_LI(NUM_BANKS, FR_BANKLIM));

	if(fill_mode == 1) {
		// zeros: load 0 into all 16 wide-data words once (never changes).
		p.add_inst(SMC_LI(0, FR_DATA));
		for(int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(FR_DATA, i));
	} else {
		p.add_inst(SMC_LI(seed, FR_STATE));
	}

	p.add_inst(SMC_LI(0, FR_BANK));               // bank = 0
	p.add_label("FILL_BANK");
	p.add_inst(SMC_LI(0, FR_ROW));                // row = 0
	p.add_label("FILL_ROW");

	if(fill_mode == 2) {
		// Per-row random: mix = state ^ row, then fill the 16 wide words with
		// successive 1-bit rotations of mix; advance state once per row.
		p.add_inst(SMC_XOR(FR_STATE, FR_ROW, FR_MIX));
		for(int i = 0; i < 16; i++) {
			p.add_inst(SMC_LDWD(FR_MIX, i));
			p.add_inst(SMC_SRC(FR_MIX, FR_MIX));
		}
		p.add_inst(SMC_ADD(FR_STATE, FR_ROW, FR_STATE));
	}

	// PRE this bank + tRP
	p.add_inst(SMC_PRE(FR_BANK, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	p.add_inst(all_nops());
	p.add_inst(all_nops());
	p.add_inst(all_nops());

	// ACT row + tRCD; irar=1 -> FR_ROW (RAR) += RASR(1) after the ACT.
	p.add_inst(SMC_ACT(FR_BANK, 0, FR_ROW, 1, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	p.add_inst(all_nops());
	p.add_inst(all_nops());
	p.add_inst(all_nops());

	// Write all NUM_COLS columns: CAR=0, icar=1 -> CAR += CASR(1) each write.
	p.add_inst(SMC_LI(0, FR_COL));
	for(int c = 0; c < NUM_COLS; c++) {
		p.add_inst(SMC_WRITE(FR_BANK, 0, FR_COL, 1, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// tWR, then PRE the bank.
	p.add_inst(all_nops());
	p.add_inst(all_nops());
	p.add_inst(SMC_PRE(FR_BANK, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	p.add_inst(all_nops());

	// FR_ROW was incremented to row+1 by the ACT; loop while it is < NUM_ROWS.
	p.add_branch(p.BR_TYPE::BL, FR_ROW, FR_ROWLIM, "FILL_ROW");

	// Next bank.
	p.add_inst(SMC_ADDI(FR_BANK, 1, FR_BANK));
	p.add_branch(p.BR_TYPE::BL, FR_BANK, FR_BANKLIM, "FILL_BANK");

	p.add_inst(SMC_END());
}

// Fill the whole DRAM of BOTH pseudo-channels (all broadcast channels) before a
// measurement. Each PC is a separate, terminating program; execute() is
// fire-and-forget over PCIe, so we sleep a safe margin (fill is ~0.1-0.2 s/PC)
// to let it finish writing before the next program reloads the instruction buffer.
void run_whole_fill(SoftMCPlatform *platform, int ch, int fill_mode, uint32_t seed) {
	for(int fpc = 0; fpc < NUM_PC; fpc++) {
		Program fill_prog;
		build_fill_program(fill_prog, ch, fpc, fill_mode, seed);
		printf("  Pre-fill: pc=%d, mode=%s, seed=0x%08x ...\n",
			fpc, fill_mode == 1 ? "zeros" : "random", seed);
		platform->execute(fill_prog);
		sleep(2); // ensure the finite fill program completes before the next execute
	}
}

void program_idd5(int ch, int pc, int bg0, int bg1) {
	// Assuming 600MHz HBM frequency => 1.67ns per cycle
	// According to JEDEC standard:
	// tRAS=33ns = 20 cycles
	// tRC=48ns = 29 cycles
	// tRP=15ns = 9 cycles
	// tRFC=350ns = 210 cycles
	// tREFI=3.9us = 2340 cycles
	int tRFC = 160;
	int tREFI = 2340;

    program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

    program.add_inst(SMC_LI(0x5555, RA5555));
    program.add_inst(SMC_LI(0x2AAA, RA2AAA));

    for(int i = 0; i < 4; i++) {
        program.add_inst(SMC_LI(4 * bg0 + i, BA0 + i));
        program.add_inst(SMC_LI(4 * bg1 + i, BA4 + i));
    }

	program.add_inst(SMC_PRE(BA0, 0, 1, pc)); // First precharge all banks
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_LI(8, BASR));

    program.add_label("IDD5_LOOP");

	for(int j = 0; j < INNER_REPS / 16; j++) {
		program.add_inst(SMC_REF(pc), SMC_NOP(), SMC_NOP(), SMC_NOP()); // Issue a refresh command to the channel
		program.add_mininst(SMC_NOP(), tRFC - 4); // Wait for tRFC
		program.pack_minprogram();
	}

    program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD5_LOOP");
}

void program_idd4r_bitflip(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Bitflip test: mask=0x%02x, bg=(%d,%d), ch=%d, pc=%d\n",
		bitflip_mask, bg0, bg1, ch, pc);
	printf("Pattern 32b:");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\n");

	// Compute which columns are negated based on bitflip mask
	// Bit i determines flip between column i and column i+1 (wraps at 8)
	// Column 0 is always original (not negated)
	bool col_negated[NUM_COLS];
	col_negated[0] = false;
	for(int c = 1; c < NUM_COLS; c++) {
		col_negated[c] = col_negated[c-1] ^ ((bitflip_mask >> ((c-1) % 8)) & 1);
	}

	printf("Column polarity: ");
	for(int c = 0; c < NUM_COLS; c++) printf("%c", col_negated[c] ? 'N' : 'O');
	printf("\n");

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(1, CASR));   // Column stride = 1 for sequential column access
	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Activate rows in all 8 banks
	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_ACT(BA0 + i, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	for(int i = 4; i < 8; i++) {
		program.add_inst(SMC_ACT(BA0 + i, 0, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Load original pattern into 512-bit register (32 bytes repeated twice)
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Write original pattern to non-negated columns (all 8 banks)
	for(int c = 0; c < NUM_COLS; c++) {
		if(!col_negated[c]) {
			program.add_inst(SMC_LI(c, CA01010));
			program.add_inst(
				SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA3, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA4, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA5, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA6, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA7, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
		}
	}

	// Load negated pattern into 512-bit register
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Write negated pattern to negated columns (all 8 banks)
	for(int c = 0; c < NUM_COLS; c++) {
		if(col_negated[c]) {
			program.add_inst(SMC_LI(c, CA01010));
			program.add_inst(
				SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA2, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA3, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA4, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA5, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA6, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
			program.add_inst(
				SMC_WRITE(BA7, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
			for(int n = 0; n < 5; n++) program.add_inst(all_nops());
		}
	}

	// READ phase: cycle through all 32 columns sequentially (infinite loop)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS / 4; j++) {
		// 4 groups of 4 instructions = 32 reads = all columns
		for(int k = 0; k < 4; k++) {
			program.add_inst(
				SMC_READ(BA0, 0, CA01010, 1, pc, 0), SMC_NOP(),
				SMC_READ(BA5, 0, CA01010, 1, pc, 0), SMC_NOP());
			program.add_inst(
				SMC_READ(BA2, 0, CA01010, 1, pc, 0), SMC_NOP(),
				SMC_READ(BA7, 0, CA01010, 1, pc, 0), SMC_NOP());
			program.add_inst(
				SMC_READ(BA1, 0, CA01010, 1, pc, 0), SMC_NOP(),
				SMC_READ(BA6, 0, CA01010, 1, pc, 0), SMC_NOP());
			program.add_inst(
				SMC_READ(BA3, 0, CA01010, 1, pc, 0), SMC_NOP(),
				SMC_READ(BA4, 0, CA01010, 1, pc, 0), SMC_NOP());
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

void program_idd4r_custom_data(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Custom data test: invert_col1=%d, bg=(%d,%d), ch=%d, pc=%d\n",
		invert_col1 ? 1 : 0, bg0, bg1, ch, pc);
	printf("Pattern 32b:");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\n");

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));

	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Activate rows in all 16 banks (j loop with BASR=8 auto-increment)
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Load original pattern into 512-bit register (32 bytes repeated twice)
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Write original pattern to col0 addresses of all 16 banks
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	if(invert_col1) {
		// Load inverted pattern for col1 addresses
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}
	// Write (same or inverted) pattern to col1 addresses of all 16 banks
	for(int j = 0; j < 16; j += 8) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	// READ phase: all 16 banks (auto-increment with BASR=8)
	program.add_label("READ_PHASE");

	for(int j = 0; j < INNER_REPS; j++) {
		program.add_inst(SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA5, 1, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA2, 1, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA7, 1, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA1, 1, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA6, 1, CA10101, 0, pc, 0), SMC_NOP());
		program.add_inst(SMC_READ(BA3, 1, CA01010, 0, pc, 0), SMC_NOP(),
				SMC_READ(BA4, 1, CA10101, 0, pc, 0), SMC_NOP());
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

void program_max_power_loop(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Max power loop: bg=(%d,%d), offset=%d, ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, bank_offsets[0], ch, pc, invert_col1 ? 1 : 0);
	printf("Pattern 32b:");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\n");

	// Load original pattern into 512-bit register (32 bytes repeated twice)
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));    // row address
	program.add_inst(SMC_LI(col_addr0, CA01010));   // column 0
	program.add_inst(SMC_LI(col_addr1, CA10101));   // column 1
	program.add_inst(SMC_LI(8, BASR));

	// Set up 2 banks: one from each bank group at the given offset
	int bank0 = 4 * bg0 + bank_offsets[0];
	int bank1 = 4 * bg1 + bank_offsets[0];
	printf("Using banks: %d (BG%d) and %d (BG%d)\n",
		bank0, bg0, bank1, bg1);
	program.add_inst(SMC_LI(bank0, BA0));
	program.add_inst(SMC_LI(bank1, BA1));

	// Precharge both banks
	program.add_inst(SMC_PRE(BA0, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_PRE(BA1, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT both banks on the same row
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_ACT(BA1, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to col0 on both banks
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to col1 on both banks
	program.add_inst(SMC_WRITE(BA0, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite read loop: alternate between col0 and col1 on both banks
	program.add_label("READ_PHASE");

	program.add_inst(all_nops());
	for(int j = 0; j < INNER_REPS * 2; j++) {
		program.add_inst(
			SMC_NOP(), SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(), SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		program.add_inst(
			SMC_NOP(), SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(), SMC_READ(BA1, 0, CA10101, 0, pc, 0));
	}
	program.add_inst(all_nops());

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// ============================================================================
// Max power loop with 3 columns per bank for 6-periodic beat patterns.
//
// Data layout (pattern_32b = A, pattern_32b_B = B, pattern_32b_C = C):
//   BA0: CA0 = A, CA1 = C, CA2 = B
//   BA1: CA0 = B, CA1 = A, CA2 = C
//
// Read order cycles: BA0,CA0 -> BA1,CA0 -> BA0,CA1 -> BA1,CA1 -> BA0,CA2 -> BA1,CA2
//
// This produces a 6-periodic beat pattern on the DQ bus when A, B, C are
// derived from a 6-bit pattern P = p0..p5:
//   A = beats p0,p1,p2,p3    (positions 0-3)
//   B = beats p4,p5,p0,p1    (positions 4-7 mod 6)
//   C = beats p2,p3,p4,p5    (positions 8-11 mod 6)
// ============================================================================
void program_max_power_loop_3col(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Max power loop 3-col: bg=(%d,%d), offset=%d, ch=%d, pc=%d\n",
		bg0, bg1, bank_offsets[0], ch, pc);
	printf("Pattern A (32b):");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\nPattern B (32b):");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b_B[i]);
	printf("\nPattern C (32b):");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b_C[i]);
	printf("\n");

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));     // row address
	program.add_inst(SMC_LI(col_addr0, CA01010));    // column 0
	program.add_inst(SMC_LI(col_addr1, CA10101));    // column 1
	program.add_inst(SMC_LI(col_addr2, CA2_REG));    // column 2
	program.add_inst(SMC_LI(8, BASR));

	// Set up 2 banks: one from each bank group at the given offset
	int bank0 = 4 * bg0 + bank_offsets[0];
	int bank1 = 4 * bg1 + bank_offsets[0];
	printf("Using banks: %d (BG%d) and %d (BG%d)\n",
		bank0, bg0, bank1, bg1);
	printf("Column addresses: CA0=0x%02x, CA1=0x%02x, CA2=0x%02x\n",
		col_addr0, col_addr1, col_addr2);
	program.add_inst(SMC_LI(bank0, BA0));
	program.add_inst(SMC_LI(bank1, BA1));

	// Precharge both banks
	program.add_inst(SMC_PRE(BA0, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_PRE(BA1, 0, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT both banks on the same row
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_ACT(BA1, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// === Write pattern A to BA0,CA0 and BA1,CA1 ===
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}
	program.add_inst(SMC_WRITE(BA0, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// === Write pattern B to BA1,CA0 and BA0,CA2 ===
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b_B[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b_B[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}
	program.add_inst(SMC_WRITE(BA1, 0, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA0, 0, CA2_REG, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	// === Write pattern C to BA0,CA1 and BA1,CA2 ===
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b_C[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b_C[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}
	program.add_inst(SMC_WRITE(BA0, 0, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA1, 0, CA2_REG, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite read loop: cycle through 3 columns on 2 banks
	// Order: BA0,CA0 -> BA1,CA0 -> BA0,CA1 -> BA1,CA1 -> BA0,CA2 -> BA1,CA2
	program.add_label("READ_PHASE");

	program.add_inst(all_nops());
	for(int j = 0; j < INNER_REPS; j++) {
		// BA0,CA0 and BA1,CA0
		program.add_inst(
			SMC_NOP(), SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(), SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// BA0,CA1 and BA1,CA1
		program.add_inst(
			SMC_NOP(), SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(), SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// BA0,CA2 and BA1,CA2
		program.add_inst(
			SMC_NOP(), SMC_READ(BA0, 0, CA2_REG, 0, pc, 0),
			SMC_NOP(), SMC_READ(BA1, 0, CA2_REG, 0, pc, 0));
	}
	program.add_inst(all_nops());

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE");
}

// ============================================================================
// Max power loop with reads on slots 1 & 3 (instead of 2 & 4).
// Otherwise identical to max_power_loop.
// ============================================================================
void program_max_power_loop_slot13(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Max power loop (slot 1&3): bg=(%d,%d), offset=%d, ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, bank_offsets[0], ch, pc, invert_col1 ? 1 : 0);
	printf("Pattern 32b:");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\n");

	// Load original pattern into 512-bit register (32 bytes repeated twice)
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));    // row address
	program.add_inst(SMC_LI(col_addr0, CA01010));   // column 0
	program.add_inst(SMC_LI(col_addr1, CA10101));   // column 1
	program.add_inst(SMC_LI(8, BASR));

	// Set up 2 banks: one from each bank group at the given offset
	int bank0 = 4 * bg0 + bank_offsets[0];
	int bank1 = 4 * bg1 + bank_offsets[0];
	program.add_inst(SMC_LI(bank0, BA0));
	program.add_inst(SMC_LI(bank1, BA1));

	// Precharge both banks
	program.add_inst(SMC_PRE(BA0, 1, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_PRE(BA0, 1, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT both banks on the same row
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to col0 on both banks
	program.add_inst(SMC_WRITE(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to col1 on both banks
	program.add_inst(SMC_WRITE(BA0, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_WRITE(BA0, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite read loop: alternate between col0 and col1 on both banks
	// Reads on slots 1 & 3 (instead of 2 & 4)
	program.add_label("READ_PHASE_SLOT13");

	program.add_inst(all_nops());
	for(int j = 0; j < INNER_REPS * 2; j++) {
		program.add_inst(
			SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0), SMC_NOP());
		program.add_inst(
			SMC_READ(BA0, 1, CA10101, 0, pc, 0), SMC_NOP(),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0), SMC_NOP());
	}
	program.add_inst(all_nops());

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "READ_PHASE_SLOT13");
}

// ============================================================================
// Max power loop across all 16 banks (IDD4R-style interleaving).
// Same data contents as max_power_loop (pattern_32b to col0, optionally
// inverted to col1) but written to all 16 banks.
// Loop: first pass reads CA01010 from all 16 banks, second reads CA10101.
// Uses BASR=8 to cover 16 banks with 8 registers.
// ============================================================================
void program_max_power_16bank(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("Max power 16-bank: bg=(%d,%d), bank_off=%d, ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, bank_offsets[0], ch, pc, invert_col1 ? 1 : 0);
	printf("Pattern 32b:");
	for(int i = 0; i < 8; i++) printf(" %08x", pattern_32b[i]);
	printf("\n");

	// Load original pattern into 512-bit register
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i));
	}
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(pattern_32b[i], PATTERN_REG));
		program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
	}

	// Set up registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to CA01010 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to CA10101 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// Registers restored after 2 passes each

	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Infinite read loop: IDD4R interleaving across all 16 banks
	// First pass: read CA01010 from all 16 banks
	// Second pass: read CA10101 from all 16 banks
	program.add_label("MAX16_READ_PHASE");

	// for(int j = 0; j < INNER_REPS / 4; j++) {
	// 	// Pass 1: CA01010 — IDD4R bank order, 2 iterations to cover 16 banks
	// 	for(int pass = 0; pass < 1; pass++) {
	// 		int ibar = 0;
	// 		program.add_inst(SMC_READ(BA0, ibar, CA01010, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA4, ibar, CA01010, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA0, ibar, CA10101, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA4, ibar, CA10101, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA1, ibar, CA01010, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA5, ibar, CA01010, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA1, ibar, CA10101, 0,pc, 0), SMC_NOP(),
	// 				SMC_READ(BA5, ibar, CA10101, 0, pc, 0), SMC_NOP());
	// 	}
	// 	// Registers restored after 2 passes (+16 ≡ 0 mod 16)

	// 	// Pass 2: CA10101 — same IDD4R order, 2 iterations to cover 16 banks
	// 	for(int pass = 0; pass < 1; pass++) {
	// 		int ibar = 0;
	// 		program.add_inst(SMC_READ(BA2, ibar, CA01010, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA6, ibar, CA01010, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA2, ibar, CA10101, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA6, ibar, CA10101, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA3, ibar, CA01010, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA7, ibar, CA01010, 0, pc, 0), SMC_NOP());
	// 		program.add_inst(SMC_READ(BA3, ibar, CA10101, 0, pc, 0), SMC_NOP(),
	// 				SMC_READ(BA7, ibar, CA10101, 0, pc, 0), SMC_NOP());
	// 	}
	// }


	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "MAX16_READ_PHASE");
}

// ============================================================================
// Variation 1: ACT all 16 banks (one every 3 cycles), then PRE-all
// Uses BASR=8: BA0-BA7 each toggle between bank X and bank X+8.
// Two passes through BA0-BA7 cover all 16 banks.
// Uses add_mininst() for 3-cycle ACT spacing (ACT + 2 NOPs).
// ============================================================================
void program_idd1_precharge_all(int ch, int pc) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 precharge_all: ch=%d, pc=%d\n", ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// Initialize BA0-BA7 = banks 0-7
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(i, BA0 + i));
	}

	// PRE-all + tRP (3 add_inst = 12 cycles > tRP=9)
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	// After 2 passes, registers return to original values (+8+8 = +16 ≡ +0 mod 16)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// tRCD (3 add_inst = 12 cycles > tRCD≈10)
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// WRITE all 16 banks (2 passes, ibar=1, registers restored after)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Main loop: ACT all 16 banks (3 cycles each), wait tRAS, PRE-all, wait tRP
	program.add_label("IDD1_PA_LOOP");

	for(int rep = 0; rep < INNER_REPS / 64; rep++) {
		// ACT 16 banks: 2 passes through BA0-BA7 (ibar=1), 3 cycles each
		// ACT + 2 NOPs = 3 cycles per bank, 16 banks = 48 cycles.
		// Last ACT at cycle 45.
		for(int pass = 0; pass < 2; pass++) {
			for(int i = 0; i < 8; i++) {
				int ROW = (i % 2) ? RA2AAA : RA5555;
				program.add_mininst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), 2);
			}
		}
		program.pack_minprogram();

		// Wait tRAS for last bank: need PRE at cycle >= 45+22=67.
		// Currently at cycle 48. Need 19 more. Use add_mininst for precise wait.
		program.add_mininst(SMC_NOP(), 18);  // 19 NOP cycles
		// PRE-all at cycle 67
		program.add_mininst(SMC_PRE(BA0, 0, 1, pc), 0);
		// tRP = 9 cycles
		program.add_mininst(SMC_NOP(), 8);
		program.pack_minprogram();
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_PA_LOOP");
}

// ============================================================================
// Variation 2a: Pipelined ACT+PRE, IDD0 bank order
// BASR=8: each register toggles between two banks.
// Init: 8 add_inst of ACT(ibar=1)+3NOPs (pipeline fill, 32 cycles).
// Steady state: ACT(ibar=1)+NOP+PRE(ibar=0)+NOP per register per add_inst.
// After ACT increments BAi by 8, PRE targets the post-increment value (bank+8),
// which is the bank that was ACT'd 8 slots = 32 cycles ago.
// tRAS = 34 cycles (8 slots × 4 + 2 offset), tRP = 30 cycles.
// ============================================================================
void program_idd1_pipelined(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 pipelined (IDD0 order): bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// IDD0 interleaved order: alternating bg0 and bg1
	// With BASR=8, each also toggles to the bank +8 (in bg0+2 / bg1+2)
	int idd0_banks[8] = {
		4*bg0+0, 4*bg1+1, 4*bg0+2, 4*bg1+3,
		4*bg0+1, 4*bg1+2, 4*bg0+3, 4*bg1+0
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(idd0_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes, ibar=1), WRITE all, PRE-all
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 8 ACTs (BA0-BA7, ibar=1), each = ACT + 3 NOPs = 1 add_inst
	// After this, each BAi holds bank_i + 8.
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// Steady state: ACT(BAi, ibar=1) + NOP + PRE(BAi, ibar=0) + NOP
	program.add_label("IDD1_PIPE_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(
				SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(),
				SMC_PRE(BA0 + i, 0, 0, pc), SMC_NOP());
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_PIPE_LOOP");
}

// ============================================================================
// Variation 2b: Pipelined ACT+PRE, bank_offset order
// Same BASR=8 trick, but registers loaded based on bank_offsets and bg0/bg1.
// ============================================================================
// ============================================================================
// Variation 2b: Pipelined ACT+PRE, bank_offset order, 8 banks only (bg0/bg1)
// Each register holds a fixed bank — no BASR=8 toggling.
// Pipeline depth D=5: ACT BA[i] and PRE BA[(i+3)%8] in same add_inst.
// tRAS = 5*4+2 = 22 cycles, tRP = 3*4-2 = 30 cycles.
// ============================================================================
void program_idd1_pipelined_bankoffset(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 pipelined (bank_offset, 8 banks): bg=(%d,%d), offsets=[%d,%d,%d,%d], ch=%d, pc=%d\n",
		bg0, bg1, bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3], ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));

	// 8 banks: interleave bg0 and bg1 with bank_offsets
	int bo_banks[8] = {
		4*bg0 + bank_offsets[0], 4*bg1 + bank_offsets[0],
		4*bg0 + bank_offsets[1], 4*bg1 + bank_offsets[1],
		4*bg0 + bank_offsets[2], 4*bg1 + bank_offsets[2],
		4*bg0 + bank_offsets[3], 4*bg1 + bank_offsets[3]
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(bo_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT 8 banks (ibar=0), WRITE, PRE-all
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_WRITE(BA0 + i, 0, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 5 ACTs without PREs (pipeline fill)
	for(int i = 0; i < 5; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// Steady state: ACT(BA[i], ibar=0) + NOP + PRE(BA[(i+3)%8], ibar=0) + NOP
	program.add_label("IDD1_PIPE_BO_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			int pre_reg = (i + 3) % 8;
			program.add_inst(
				SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(),
				SMC_PRE(BA0 + pre_reg, 0, 0, pc), SMC_NOP());
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_PIPE_BO_LOOP");
}

// ============================================================================
// Variation 3a: Pipelined ACT+RDAP, IDD0 bank order
// Same BASR=8 trick. RDAP = READ with ap=1 (auto-precharge).
// Steady state: ACT(ibar=1) + NOP + RDAP(ibar=0) + NOP
// tRAS = 34 cycles, tRCD = 34 cycles (from 8 slots ago to RDAP).
// ============================================================================
void program_idd1_rdap(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 RDAP (IDD0 order): bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// IDD0 interleaved order
	int idd0_banks[8] = {
		4*bg0+0, 4*bg1+1, 4*bg0+2, 4*bg1+3,
		4*bg0+1, 4*bg1+2, 4*bg0+3, 4*bg1+0
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(idd0_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes, ibar=1), WRITE all, PRE-all
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 8 ACTs (BA0-BA7, ibar=1)
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// Steady state: ACT(BAi, ibar=1) + NOP + RDAP(BAi, ibar=0) + NOP
	program.add_label("IDD1_RDAP_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(
				SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(),
				SMC_NOP(), SMC_READ(BA0 + i, 0, CA00000, 0, pc, 1));
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_RDAP_LOOP");
}

// ============================================================================
// Variation 3a single-pass: same as program_idd1_rdap but runs only once
// (2 reps of the 8-bank pattern, no outer loop).
// ============================================================================
void program_idd1_rdap_singlepass(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 RDAP single-pass (IDD0 order): bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// IDD0 interleaved order
	int idd0_banks[8] = {
		4*bg0+0, 4*bg1+1, 4*bg0+2, 4*bg1+3,
		4*bg0+1, 4*bg1+2, 4*bg0+3, 4*bg1+0
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(idd0_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes, ibar=1), WRITE all, PRE-all
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 8 ACTs (BA0-BA7, ibar=1)
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_mininst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), 3);
	}

	// Single pass: 2 reps of the 8-bank pattern, no outer loop
	for(int rep = 0; rep < 2; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_mininst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), 2);
			program.add_mininst(SMC_READ(BA0 + i, 0, CA00000, 0, pc, 1), 0);
		}
	}

	program.pack_minprogram();
}

// ============================================================================
// IDD7: IDD4R-based loop with one ACT every 6 cycles.
// Reads on slots 2 & 4 of each add_inst, ACTs on slots 1 & 3.
// 24 add_inst per iteration (2 phases of 12):
//   Phase A: read BA0,BA5,BA2,BA7 / ACT BA1,BA6,BA3,BA4
//   Phase B: read BA1,BA6,BA3,BA4 / ACT BA0,BA5,BA2,BA7 (swapped)
// Each phase: 3 sets of 4 add_inst. Set 3 last 2 use RDAP (ap=1).
// ============================================================================
void program_idd7(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7: bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// After 2 passes, registers restored (+16 ≡ 0 mod 16)
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// Registers restored after 2 passes

	// Precharge all banks, then ACT only the 8 read banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT 8 read banks: first pass ibar=0 (4 banks), second pass ibar=1 (+8, 4 more banks)
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// Undo ibar=1 increment on read registers

	// Steady-state loop: 24 add_inst per iteration
	// First 12: read BA0,BA5,BA2,BA7 / ACT BA1,BA6,BA3,BA4
	// Second 12: read BA1,BA6,BA3,BA4 / ACT BA0,BA5,BA2,BA7 (swapped)
	program.add_label("IDD7_LOOP");

	for(int j = 0; j < INNER_REPS / 24; j++) {
		// ============================================================
		// Phase A: read BA0,BA5,BA2,BA7 / ACT BA1,BA6,BA3,BA4
		// ============================================================

		// === Set 1 (4 add_inst) ===
		// inst 1: ACT(BA1,ibar=1), RD(BA0), NOP, RD(BA5)
		program.add_inst(
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0));
		// inst 2: NOP, RD(BA2), ACT(BA6,ibar=1), RD(BA7)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA2, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0));
		// inst 3: NOP, RD(BA0), NOP, RD(BA5)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0));
		// inst 4: ACT(BA3,ibar=1), RD(BA2), NOP, RD(BA7)
		program.add_inst(
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_READ(BA2, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0));

		// === Set 2 (4 add_inst) ===
		// inst 5: NOP, RD(BA0), ACT(BA4,ibar=1), RD(BA5)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0));
		// inst 6: NOP, RD(BA2), NOP, RD(BA7)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA2, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0));
		// inst 7: ACT(BA1,ibar=1), RD(BA0), NOP, RD(BA5)
		program.add_inst(
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 0));
		// inst 8: NOP, RD(BA2), ACT(BA6,ibar=1), RD(BA7)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA2, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA7, 1, CA10101, 0, pc, 0));

		// === Set 3 (4 add_inst, RDAP on BA0,BA5,BA2,BA7) ===
		// inst 9: NOP, RD(BA0), NOP, RD(BA5)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA5, 1, CA10101, 0, pc, 1));
		// inst 10: ACT(BA3,ibar=1), RD(BA2), NOP, RD(BA7)
		program.add_inst(
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_READ(BA2, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 1));
		// inst 11: NOP, RDAP(BA0,ibar=1), ACT(BA4,ibar=1), RDAP(BA5,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 1),
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_READ(BA5, 1, CA10101, 0, pc, 1));
		// inst 12: NOP, RDAP(BA2,ibar=1), NOP, RDAP(BA7,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA2, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA7, 1, CA10101, 0, pc, 1));

		// ============================================================
		// Phase B (swapped): read BA1,BA6,BA3,BA4 / ACT BA0,BA5,BA2,BA7
		// ============================================================

		// === Set 4 (4 add_inst) ===
		// inst 13: ACT(BA0,ibar=1), RD(BA1), NOP, RD(BA6)
		program.add_inst(
			SMC_ACT(BA0, 1, RA5555, 0, pc),
			SMC_READ(BA1, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0));
		// inst 14: NOP, RD(BA3), ACT(BA5,ibar=1), RD(BA4)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA3, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0));
		// inst 15: NOP, RD(BA1), NOP, RD(BA6)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA1, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0));
		// inst 16: ACT(BA2,ibar=1), RD(BA3), NOP, RD(BA4)
		program.add_inst(
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_READ(BA3, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0));

		// === Set 5 (4 add_inst) ===
		// inst 17: NOP, RD(BA1), ACT(BA7,ibar=1), RD(BA6)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA1, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0));
		// inst 18: NOP, RD(BA3), NOP, RD(BA4)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA3, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0));
		// inst 19: ACT(BA0,ibar=1), RD(BA1), NOP, RD(BA6)
		program.add_inst(
			SMC_ACT(BA0, 1, RA5555, 0, pc),
			SMC_READ(BA1, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 0));
		// inst 20: NOP, RD(BA3), ACT(BA5,ibar=1), RD(BA4)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA3, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA4, 1, CA10101, 0, pc, 0));

		// === Set 6 (4 add_inst, RDAP on BA1,BA6,BA3,BA4) ===
		// inst 21: NOP, RD(BA1), NOP, RD(BA6)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA1, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA6, 1, CA10101, 0, pc, 1));
		// inst 22: ACT(BA2,ibar=1), RD(BA3), NOP, RD(BA4)
		program.add_inst(
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_READ(BA3, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 1));
		// inst 23: NOP, RDAP(BA1,ibar=1), ACT(BA7,ibar=1), RDAP(BA6,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA1, 1, CA01010, 0, pc, 1),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA6, 1, CA10101, 0, pc, 1));
		// inst 24: NOP, RDAP(BA3,ibar=1), NOP, RDAP(BA4,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA3, 1, CA01010, 0, pc, 1),
			SMC_NOP(),
			SMC_READ(BA4, 1, CA10101, 0, pc, 1));
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_LOOP");
}

// ============================================================================
// IDD7 no-read: identical to IDD7 but without reads.
// Regular READs (ap=0) replaced with NOPs.
// RDAPs (ap=1) replaced with explicit PREs to maintain bank open/close timing.
// ACTs remain unchanged.
// ============================================================================
void program_idd7_noread(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7 no-read: bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// After 2 passes, registers restored (+16 ≡ 0 mod 16)
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// WRITE all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
		for(int i = 4; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	// Registers restored after 2 passes

	// Precharge all banks, then ACT only the 8 read banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT 8 read banks: first pass ibar=0 (4 banks), second pass ibar=1 (+8, 4 more banks)
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// Undo ibar=1 increment on read registers

	// Steady-state loop: 24 add_inst per iteration
	// Same structure as IDD7 but READs replaced with NOPs, RDAPs with PREs
	program.add_label("IDD7_NOREAD_LOOP");

	for(int j = 0; j < INNER_REPS / 24; j++) {
		// ============================================================
		// Phase A: PRE BA0,BA5,BA2,BA7 / ACT BA1,BA6,BA3,BA4
		// ============================================================

		// === Set 1 (4 add_inst) ===
		// inst 1: ACT(BA1,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 2: NOP, NOP, ACT(BA6,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_NOP());
		// inst 3: NOP, NOP, NOP, NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 4: ACT(BA3,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());

		// === Set 2 (4 add_inst) ===
		// inst 5: NOP, NOP, ACT(BA4,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_NOP());
		// inst 6: NOP, NOP, NOP, NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 7: ACT(BA1,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 8: NOP, NOP, ACT(BA6,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_NOP());

		// === Set 3 (4 add_inst, PRE on BA0,BA5,BA2,BA7 replacing RDAP) ===
		// inst 9: NOP, PRE(BA0,ibar=1), NOP, PRE(BA5,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA0, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA5, 1, 0, pc));
		// inst 10: ACT(BA3,ibar=1), PRE(BA2,ibar=1), NOP, PRE(BA7,ibar=1)
		program.add_inst(
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_PRE(BA2, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA7, 1, 0, pc));
		// inst 11: NOP, PRE(BA0,ibar=1), ACT(BA4,ibar=1), PRE(BA5,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA0, 1, 0, pc),
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_PRE(BA5, 1, 0, pc));
		// inst 12: NOP, PRE(BA2,ibar=1), NOP, PRE(BA7,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA2, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA7, 1, 0, pc));

		// ============================================================
		// Phase B (swapped): PRE BA1,BA6,BA3,BA4 / ACT BA0,BA5,BA2,BA7
		// ============================================================

		// === Set 4 (4 add_inst) ===
		// inst 13: ACT(BA0,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA0, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 14: NOP, NOP, ACT(BA5,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_NOP());
		// inst 15: NOP, NOP, NOP, NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 16: ACT(BA2,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());

		// === Set 5 (4 add_inst) ===
		// inst 17: NOP, NOP, ACT(BA7,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_NOP());
		// inst 18: NOP, NOP, NOP, NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 19: ACT(BA0,ibar=1), NOP, NOP, NOP
		program.add_inst(
			SMC_ACT(BA0, 1, RA5555, 0, pc),
			SMC_NOP(),
			SMC_NOP(),
			SMC_NOP());
		// inst 20: NOP, NOP, ACT(BA5,ibar=1), NOP
		program.add_inst(
			SMC_NOP(),
			SMC_NOP(),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_NOP());

		// === Set 6 (4 add_inst, PRE on BA1,BA6,BA3,BA4 replacing RDAP) ===
		// inst 21: NOP, PRE(BA1,ibar=1), NOP, PRE(BA6,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA1, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA6, 1, 0, pc));
		// inst 22: ACT(BA2,ibar=1), PRE(BA3,ibar=1), NOP, PRE(BA4,ibar=1)
		program.add_inst(
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_PRE(BA3, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA4, 1, 0, pc));
		// inst 23: NOP, PRE(BA1,ibar=1), ACT(BA7,ibar=1), PRE(BA6,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA1, 1, 0, pc),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_PRE(BA6, 1, 0, pc));
		// inst 24: NOP, PRE(BA3,ibar=1), NOP, PRE(BA4,ibar=1)
		program.add_inst(
			SMC_NOP(),
			SMC_PRE(BA3, 1, 0, pc),
			SMC_NOP(),
			SMC_PRE(BA4, 1, 0, pc));
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_NOREAD_LOOP");
}

// ============================================================================
// IDD7 ACT/PRE + max-power reads:
// Combines the ACT/PRE bank-cycling pattern (7 bank registers, 6-cycle spacing)
// with continuous reads on 2 banks (BA0, ibar=1 → banks 0 and 8).
//
// Slots 2 & 4: reads on BA0 (ibar=1), alternating CA01010 and CA10101
// Slots 1 & 3: ACTs and PREs on BA5,BA2,BA7,BA1,BA6,BA3,BA4
//   - ACTs use ibar=1 (increment register by 8)
//   - PREs use ibar=0 (no increment), targeting the +8 bank
//   - 6 slots between consecutive ACTs, PRE 2 slots after each ACT
//
// 21 add_inst per loop iteration (2 passes through 7 ACT banks).
// ============================================================================
void program_idd7_actpre_maxpower(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7 ACT/PRE + max-power reads: bg=(%d,%d), ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, ch, pc, invert_col1 ? 1 : 0);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	// Load original pattern into 512-bit register
	load_pattern_to_program();

	// Set up address registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	// Set up bank registers: BA0-BA3 = bg0, BA5-BA7 = bg1
	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}
	// Override BA4 = BA0 + 8 for alternating reads
	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0] + 8, BA4));

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		for(int i = 4; i < 8; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to CA01010 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to CA10101 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	// Registers restored after 2 passes each

	// Precharge all, then ACT the 2 read banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT bank 0 (BA0, no ibar)
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	// ACT bank 8 (BA4, no ibar)
	program.add_inst(SMC_ACT(BA4, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT the +8 banks for BA1-BA3,BA5-BA7 so the first PREs in the loop don't hit closed banks.
	// Each ACT uses ibar=1, which increments the register by 8.
	// After each ACT the register points to the +8 bank; we need it back to the original,
	// so we ACT again (covering the base bank) to restore: +8 +8 = +16 ≡ 0 mod 16.
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA1, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA6, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA3, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// Now BA1-BA3,BA5-BA7 point to the +8 banks (incremented once).
	// ACT again to cover the base banks and restore registers (+16 ≡ 0 mod 16).
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA2, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA1, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA6, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA3, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// BA1-BA3,BA5-BA7 restored to original values; all 12 banks (base and +8) are now activated.
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Steady-state loop: 18 add_inst per iteration (2 passes × 6 ACT banks)
	// Slots 2,4: RD alternating BA0/BA4, ibar=0
	// Slots 1,3: ACTs (ibar=1) and PREs (ibar=0), 6-slot ACT spacing, PRE 2 slots after ACT
	program.add_label("IDD7_ACTPRE_MAXPOWER_LOOP");

	for(int j = 0; j < INNER_REPS / 18; j++) {
		// === Pass 1: ACTs starting on slot 1 ===

		// inst 0: ACT(BA5,ibar=1)@s1, RD(BA0,col0)@s2, PRE(BA2,ibar=0)@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_PRE(BA2, 0, 0, pc),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 1: NOP@s1, RD(BA4,col1)@s2, ACT(BA2,ibar=1)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 2: PRE(BA7,ibar=0)@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_PRE(BA7, 0, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 3: ACT(BA7,ibar=1)@s1, RD(BA4,col1)@s2, PRE(BA1,ibar=0)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_PRE(BA1, 0, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 4: NOP@s1, RD(BA0,col0)@s2, ACT(BA1,ibar=1)@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 5: PRE(BA6,ibar=0)@s1, RD(BA4,col1)@s2, NOP@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_PRE(BA6, 0, 0, pc),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 6: ACT(BA6,ibar=1)@s1, RD(BA0,col0)@s2, PRE(BA3,ibar=0)@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_PRE(BA3, 0, 0, pc),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 7: NOP@s1, RD(BA4,col1)@s2, ACT(BA3,ibar=1)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 8: PRE(BA5,ibar=0)@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_PRE(BA5, 0, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));

		// === Pass 2: ACTs starting on slot 3 ===

		// inst 9: NOP@s1, RD(BA4,col1)@s2, ACT(BA5,ibar=1)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 10: PRE(BA2,ibar=0)@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_PRE(BA2, 0, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 11: ACT(BA2,ibar=1)@s1, RD(BA4,col1)@s2, PRE(BA7,ibar=0)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_ACT(BA2, 1, RA5555, 0, pc),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_PRE(BA7, 0, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 12: NOP@s1, RD(BA0,col0)@s2, ACT(BA7,ibar=1)@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 13: PRE(BA1,ibar=0)@s1, RD(BA4,col1)@s2, NOP@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_PRE(BA1, 0, 0, pc),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 14: ACT(BA1,ibar=1)@s1, RD(BA0,col0)@s2, PRE(BA6,ibar=0)@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_ACT(BA1, 1, RA5555, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_PRE(BA6, 0, 0, pc),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 15: NOP@s1, RD(BA4,col1)@s2, ACT(BA6,ibar=1)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
		// inst 16: PRE(BA3,ibar=0)@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_PRE(BA3, 0, 0, pc),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA4, 0, CA01010, 0, pc, 0));
		// inst 17: ACT(BA3,ibar=1)@s1, RD(BA4,col1)@s2, PRE(BA5,ibar=0)@s3, RD(BA4,col1)@s4
		program.add_inst(
			SMC_ACT(BA3, 1, RA5555, 0, pc),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_PRE(BA5, 0, 0, pc),
			SMC_READ(BA4, 0, CA10101, 0, pc, 0));
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_ACTPRE_MAXPOWER_LOOP");
}

// ============================================================================
// IDD7 max-power reads only (no ACT/PRE):
// Identical structure to idd7_actpre_maxpower but with all ACTs and PREs in
// slots 1 & 3 replaced by NOPs. Only the reads on BA0 (ibar=1) remain.
//
// 21 add_inst per loop iteration to match timing of the full variant.
// ============================================================================
void program_idd7_maxpower_rdonly(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7 max-power reads only: bg=(%d,%d), ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, ch, pc, invert_col1 ? 1 : 0);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	// Load original pattern into 512-bit register
	load_pattern_to_program();

	// Set up address registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	// Set up bank registers: BA0-BA3 = bg0, BA4-BA7 = bg1
	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		for(int i = 4; i < 8; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to CA01010 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to CA10101 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	// Registers restored after 2 passes each

	// Precharge all, then ACT the 2 read banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Set BA1 = BA0 + 8 for alternating reads
	program.add_inst(SMC_LI(4 * bg0 + bank_offsets[0] + 8, BA1));

	// ACT bank 0 (BA0, no ibar)
	program.add_inst(SMC_ACT(BA0, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	// ACT bank 8 (BA1, no ibar)
	program.add_inst(SMC_ACT(BA1, 0, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Steady-state loop: 21 add_inst per iteration (matching actpre_maxpower timing)
	// Slots 2,4: RD alternating BA0/BA1, ibar=0
	// Slots 1,3: NOPs (no ACTs or PREs)
	program.add_label("IDD7_MAXPOWER_RDONLY_LOOP");

	for(int j = 0; j < INNER_REPS / 21; j++) {
		// === Pass 1 ===

		// inst 0: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 1: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 2: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 3: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 4: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 5: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 6: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 7: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 8: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 9: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));

		// === Pass 2 ===

		// inst 10: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 11: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 12: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 13: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 14: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 15: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 16: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 17: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 18: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
		// inst 19: NOP@s1, RD(BA1,col1)@s2, NOP@s3, RD(BA1,col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA10101, 0, pc, 0));
		// inst 20: NOP@s1, RD(BA0,col0)@s2, NOP@s3, RD(BA0,col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 0, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA1, 0, CA01010, 0, pc, 0));
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_MAXPOWER_RDONLY_LOOP");
}

// ============================================================================
// IDD7 ACT/PRE (4 regs) + max-power reads:
// Same as idd7_actpre_maxpower but only ACTs/PREs the 8 banks covered by
// BA4-BA7 (instead of BA1-BA7). Reads on BA0 (ibar=1) remain identical.
//
// Slots 2 & 4: reads on BA0 (ibar=1), same-column per instruction, alternating
// Slots 1 & 3: ACTs and PREs on BA5,BA7,BA6,BA4
//   - ACTs use ibar=1; PREs use ibar=0
//   - 6 slots between consecutive ACTs, PRE 2 slots after each ACT
//
// 12 add_inst per loop iteration (2 passes through 4 ACT banks).
// ============================================================================
void program_idd7_actpre_maxpower_half(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7 ACT/PRE (4 regs) + max-power reads: bg=(%d,%d), ch=%d, pc=%d, invert_col1=%d\n",
		bg0, bg1, ch, pc, invert_col1 ? 1 : 0);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	// Load original pattern into 512-bit register
	load_pattern_to_program();

	// Set up address registers
	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA01010));
	program.add_inst(SMC_LI(col_addr1, CA10101));
	program.add_inst(SMC_LI(8, BASR));

	// Set up bank registers: BA0-BA3 = bg0, BA4-BA7 = bg1
	for(int i = 0; i < 4; i++) {
		program.add_inst(SMC_LI(4 * bg0 + bank_offsets[i], BA0 + i));
		program.add_inst(SMC_LI(4 * bg1 + bank_offsets[i], BA4 + i));
	}

	// Precharge all banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT all 16 banks (2 passes through BA0-BA7, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 4; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		for(int i = 4; i < 8; i++)
			program.add_inst(SMC_ACT(BA0 + i, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Write original pattern to CA01010 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA01010, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	if(invert_col1) {
		// Load inverted pattern for col1
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i));
		}
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_LI(~pattern_32b[i], PATTERN_REG));
			program.add_inst(SMC_LDWD(PATTERN_REG, i + 8));
		}
	}

	// Write (same or inverted) pattern to CA10101 on all 16 banks (2 passes, ibar=1)
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++)
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA10101, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	// Registers restored after 2 passes each

	// Precharge all, then ACT the 2 read banks
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT bank 0 (BA0=0, ibar=1 → BA0 becomes 8)
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	// ACT bank 8 (BA0=8, ibar=1 → BA0 becomes 0, restored)
	program.add_inst(SMC_ACT(BA0, 1, RA5555, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// ACT the +8 banks for BA4-BA7 so the first PREs in the loop don't hit closed banks.
	// Each ACT uses ibar=1. Two passes to restore registers (+16 ≡ 0 mod 16).
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA6, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA4, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// Now BA4-BA7 point to the +8 banks. ACT again to cover base and restore.
	program.add_inst(SMC_ACT(BA5, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA7, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA6, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	program.add_inst(SMC_ACT(BA4, 1, RA2AAA, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	program.add_inst(all_nops());
	// BA4-BA7 restored; all 8 banks (base and +8) are now activated.
	program.add_inst(all_nops());
	program.add_inst(all_nops());
	program.add_inst(all_nops());

	// Steady-state loop: 12 add_inst per iteration (2 passes × 4 ACT banks)
	// Slots 2,4: RD(BA0, ibar=1, col) on every instruction
	// Slots 1,3: ACTs (ibar=1) and PREs (ibar=0), 6-slot ACT spacing, PRE 2 slots after ACT
	// Bank order: BA5, BA7, BA6, BA4
	program.add_label("IDD7_ACTPRE_MAXPOWER_HALF_LOOP");

	for(int j = 0; j < INNER_REPS / 12; j++) {
		// === Pass 1: ACTs starting on slot 1 ===

		// inst 0: ACT(BA5,ibar=1)@s1, RD(col0)@s2, PRE(BA5,ibar=0)@s3, RD(col0)@s4
		program.add_inst(
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_PRE(BA5, 0, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 1: NOP@s1, RD(col1)@s2, ACT(BA7,ibar=1)@s3, RD(col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));
		// inst 2: PRE(BA7,ibar=0)@s1, RD(col0)@s2, NOP@s3, RD(col0)@s4
		program.add_inst(
			SMC_PRE(BA7, 0, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 3: ACT(BA6,ibar=1)@s1, RD(col1)@s2, PRE(BA6,ibar=0)@s3, RD(col1)@s4
		program.add_inst(
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_PRE(BA6, 0, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));
		// inst 4: NOP@s1, RD(col0)@s2, ACT(BA4,ibar=1)@s3, RD(col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 5: PRE(BA4,ibar=0)@s1, RD(col1)@s2, NOP@s3, RD(col1)@s4
		program.add_inst(
			SMC_PRE(BA4, 0, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));

		// === Pass 2: ACTs starting on slot 3 ===

		// inst 6: NOP@s1, RD(col0)@s2, ACT(BA5,ibar=1)@s3, RD(col0)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_ACT(BA5, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 7: PRE(BA5,ibar=0)@s1, RD(col1)@s2, NOP@s3, RD(col1)@s4
		program.add_inst(
			SMC_PRE(BA5, 0, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));
		// inst 8: ACT(BA7,ibar=1)@s1, RD(col0)@s2, PRE(BA7,ibar=0)@s3, RD(col0)@s4
		program.add_inst(
			SMC_ACT(BA7, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_PRE(BA7, 0, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 9: NOP@s1, RD(col1)@s2, ACT(BA6,ibar=1)@s3, RD(col1)@s4
		program.add_inst(
			SMC_NOP(),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_ACT(BA6, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));
		// inst 10: PRE(BA6,ibar=0)@s1, RD(col0)@s2, NOP@s3, RD(col0)@s4
		program.add_inst(
			SMC_PRE(BA6, 0, 0, pc),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0),
			SMC_NOP(),
			SMC_READ(BA0, 1, CA01010, 0, pc, 0));
		// inst 11: ACT(BA4,ibar=1)@s1, RD(col1)@s2, PRE(BA4,ibar=0)@s3, RD(col1)@s4
		program.add_inst(
			SMC_ACT(BA4, 1, RA2AAA, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0),
			SMC_PRE(BA4, 0, 0, pc),
			SMC_READ(BA0, 1, CA10101, 0, pc, 0));
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_ACTPRE_MAXPOWER_HALF_LOOP");
}

// ============================================================================
// IDD7 experimentation: based on Variation 3a (Pipelined ACT+RDAP, IDD0 order)
// Changes vs 3a:
//   - SMC_READ with auto-precharge (ap=1) moved to 4th slot
//   - Additional SMC_READ without auto-precharge (ap=0) in 2nd slot
// Steady state: ACT(ibar=1) + RD(ap=0) + NOP + RDAP(ap=1)
// ============================================================================
void program_idd7_experimental(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD7 experimental (IDD0 order): bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// IDD0 interleaved order
	int idd0_banks[8] = {
		4*bg0+0, 4*bg1+1, 4*bg0+2, 4*bg1+3,
		4*bg0+1, 4*bg1+2, 4*bg0+3, 4*bg1+0
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(idd0_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes, ibar=1), WRITE all, PRE-all
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 8 ACTs (BA0-BA7, ibar=1)
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// Steady state: ACT(BAi, ibar=1) + RD(BAi, ap=0) + NOP + RDAP(BAi, ap=1)
	program.add_label("IDD7_EXPERIMENTAL_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(
				SMC_ACT(BA0 + i, 1, ROW, 0, pc),
				SMC_READ(BA0 + i, 0, CA00000, 0, pc, 0),
				SMC_NOP(),
				SMC_READ(BA0 + i, 0, CA00000, 0, pc, 1));
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD7_EXPERIMENTAL_LOOP");
}

// ============================================================================
// Variation 3b: Pipelined ACT+RDAP, bank_offset order
// Same BASR=8 trick with RDAP, registers loaded based on bank_offsets.
// ============================================================================
// ============================================================================
// Variation 3b: Pipelined ACT+RDAP, bank_offset order, 8 banks only (bg0/bg1)
// Each register holds a fixed bank — no BASR=8 toggling.
// Pipeline depth D=5: ACT BA[i] and RDAP BA[(i+3)%8] in same add_inst.
// tRAS = 5*4+2 = 22 cycles, tRP(from RDAP) = 3*4-2 = 30 cycles.
// ============================================================================
void program_idd1_rdap_bankoffset(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 RDAP (bank_offset, 8 banks): bg=(%d,%d), offsets=[%d,%d,%d,%d], ch=%d, pc=%d\n",
		bg0, bg1, bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3], ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));

	// 8 banks: interleave bg0 and bg1 with bank_offsets
	int bo_banks[8] = {
		4*bg0 + bank_offsets[0], 4*bg1 + bank_offsets[0],
		4*bg0 + bank_offsets[1], 4*bg1 + bank_offsets[1],
		4*bg0 + bank_offsets[2], 4*bg1 + bank_offsets[2],
		4*bg0 + bank_offsets[3], 4*bg1 + bank_offsets[3]
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(bo_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT 8 banks (ibar=0), WRITE, PRE-all
	for(int i = 0; i < 8; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_WRITE(BA0 + i, 0, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 5 ACTs without RDAPs (pipeline fill)
	for(int i = 0; i < 5; i++) {
		int ROW = (i % 2) ? RA2AAA : RA5555;
		program.add_inst(SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
	}

	// Steady state: ACT(BA[i], ibar=0) + NOP + RDAP(BA[(i+3)%8], ibar=0) + NOP
	program.add_label("IDD1_RDAP_BO_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			int rdap_reg = (i + 3) % 8;
			program.add_inst(
				SMC_ACT(BA0 + i, 0, ROW, 0, pc), SMC_NOP(),
				SMC_READ(BA0 + rdap_reg, 0, CA00000, 0, pc, 1), SMC_NOP());
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_RDAP_BO_LOOP");
}

// Variation 4: maximize ACT+PRE throughput with 3-cycle ACT spacing
// Pattern per 3 add_insts (12 cycles, 4 ACTs + 4 PREs):
//   ACT(BAx,  i1) NOP            NOP              ACT(BAx+1,i1)
//   PRE(BAx,  i0) PRE(BAx+1,i0) ACT(BAx+2,i1)   NOP
//   NOP           ACT(BAx+3,i1) PRE(BAx+2,i0)   PRE(BAx+3,i0)
// Uses BASR=8 toggling (16 banks across all 4 bank groups).
// First 2 groups (8 ACTs) are init with no PREs.
void program_idd1_max_actpre(int ch, int pc, int bg0, int bg1) {
	program.add_inst(SMC_SEL_CH(ch, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());

	printf("IDD1 max ACT+PRE (3-cycle spacing): bg=(%d,%d), ch=%d, pc=%d\n", bg0, bg1, ch, pc);
	printf("Multi-channel: %d, pattern: %x\n", MULTI_CHANNEL, wr_pattern);

	load_pattern_to_program();

	program.add_inst(SMC_LI(row_addr0, RA5555));
	program.add_inst(SMC_LI(row_addr1, RA2AAA));
	program.add_inst(SMC_LI(col_addr0, CA00000));
	program.add_inst(SMC_LI(8, BASR));

	// IDD0 interleaved order: alternating bg0 and bg1
	// With BASR=8, each also toggles to the bank +8 (in bg0+2 / bg1+2)
	int idd0_banks[8] = {
		4*bg0+0, 4*bg1+1, 4*bg0+2, 4*bg1+3,
		4*bg0+1, 4*bg1+2, 4*bg0+3, 4*bg1+0
	};
	for(int i = 0; i < 8; i++) {
		program.add_inst(SMC_LI(idd0_banks[i], BA0 + i));
	}

	// PRE-all + tRP
	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init write: ACT all 16 banks (2 passes, ibar=1), WRITE all, PRE-all
	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			int ROW = (i % 2) ? RA2AAA : RA5555;
			program.add_inst(SMC_ACT(BA0 + i, 1, ROW, 0, pc), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}
	for(int i = 0; i < 3; i++) program.add_inst(all_nops()); // tRCD

	for(int pass = 0; pass < 2; pass++) {
		for(int i = 0; i < 8; i++) {
			program.add_inst(SMC_WRITE(BA0 + i, 1, CA00000, 0, pc, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());
		}
	}

	program.add_inst(SMC_PRE(BA0, 0, 1, pc));
	for(int i = 0; i < 3; i++) program.add_inst(all_nops());

	// Init phase: 2 groups of 4 ACTs (no PREs) to fill all 8 registers
	// Group 0: BA0-BA3
	{
		int x = 0;
		program.add_inst(
			SMC_ACT(BA0 + x, 1, RA5555, 0, pc), SMC_NOP(),
			SMC_NOP(), SMC_ACT(BA0 + x + 1, 1, RA2AAA, 0, pc));
		program.add_inst(
			SMC_NOP(), SMC_NOP(),
			SMC_ACT(BA0 + x + 2, 1, RA5555, 0, pc), SMC_NOP());
		program.add_inst(
			SMC_NOP(), SMC_ACT(BA0 + x + 3, 1, RA2AAA, 0, pc),
			SMC_NOP(), SMC_NOP());
	}
	// Group 1: BA4-BA7
	{
		int x = 4;
		program.add_inst(
			SMC_ACT(BA0 + x, 1, RA5555, 0, pc), SMC_NOP(),
			SMC_NOP(), SMC_ACT(BA0 + x + 1, 1, RA2AAA, 0, pc));
		program.add_inst(
			SMC_NOP(), SMC_NOP(),
			SMC_ACT(BA0 + x + 2, 1, RA5555, 0, pc), SMC_NOP());
		program.add_inst(
			SMC_NOP(), SMC_ACT(BA0 + x + 3, 1, RA2AAA, 0, pc),
			SMC_NOP(), SMC_NOP());
	}

	// Steady state: alternating groups of BA0-BA3 and BA4-BA7
	program.add_label("IDD1_MAX_ACTPRE_LOOP");

	for(int rep = 0; rep < INNER_REPS / 8; rep++) {
		for(int g = 0; g < 2; g++) {
			int x = g * 4;
			// ACT(BAx,i1) NOP NOP ACT(BAx+1,i1)
			program.add_inst(
				SMC_ACT(BA0 + x, 1, RA5555, 0, pc), SMC_NOP(),
				SMC_NOP(), SMC_ACT(BA0 + x + 1, 1, RA2AAA, 0, pc));
			// PRE(BAx,i0) PRE(BAx+1,i0) ACT(BAx+2,i1) NOP
			program.add_inst(
				SMC_PRE(BA0 + x, 0, 0, pc), SMC_PRE(BA0 + x + 1, 0, 0, pc),
				SMC_ACT(BA0 + x + 2, 1, RA5555, 0, pc), SMC_NOP());
			// NOP ACT(BAx+3,i1) PRE(BAx+2,i0) PRE(BAx+3,i0)
			program.add_inst(
				SMC_NOP(), SMC_ACT(BA0 + x + 3, 1, RA2AAA, 0, pc),
				SMC_PRE(BA0 + x + 2, 0, 0, pc), SMC_PRE(BA0 + x + 3, 0, 0, pc));
		}
	}

	program.add_branch(program.BR_TYPE::JUMP, 0, 0, "IDD1_MAX_ACTPRE_LOOP");
}

void power_measurement(SoftMCPlatform *platform, int dimm_select, int ch, int pc, int bg0, int bg1)
{
	
	printf("Starting power measurement for bg pair (%d, %d) on channel %d, pc %d\n", bg0, bg1, ch, pc);
	//SoftMCPlatform platform(false);

	if(PATTERN == 0) { wr_pattern = zeros; shift_pattern = true; use_multi_word_pattern = false; }
	else if (PATTERN == 1) { wr_pattern = ones; shift_pattern = true; use_multi_word_pattern = false; }
	else if (PATTERN == 2) { wr_pattern = idd4r; shift_pattern = true; use_multi_word_pattern = false; }
	else if (PATTERN == 3) { wr_pattern = all_as; shift_pattern = true; use_multi_word_pattern = false; }
	else if (PATTERN == 4) { wr_pattern = custom_wr_pattern; shift_pattern = false; use_multi_word_pattern = false; }
	else if (PATTERN == 5) { wr_pattern = custom_wr_pattern; shift_pattern = true; use_multi_word_pattern = false; }
	else if (PATTERN == 6) {
		// 64b 0x00, 64b 0x55, 64b 0xFF, 64b 0xAA — repeated twice (512 bits)
		use_multi_word_pattern = true;
		uint32_t seq[8] = {
			0x00000000, 0x00000000, // 64 bits of 0x00
			0x55555555, 0x55555555, // 64 bits of 0x55
			0xFFFFFFFF, 0xFFFFFFFF, // 64 bits of 0xFF
			0xAAAAAAAA, 0xAAAAAAAA  // 64 bits of 0xAA
		};
		for(int i = 0; i < 8; i++) {
			multi_word_pattern[i] = seq[i];
			multi_word_pattern[i + 8] = seq[i];
		}
		wr_pattern = multi_word_pattern[0]; // for printf
	}
	else if (PATTERN == 7) {
		// 32-byte pattern (8 x uint32_t), loaded twice into 512-bit register
		use_multi_word_pattern = true;
		for(int i = 0; i < 8; i++) {
			multi_word_pattern[i] = pattern_32b[i];
			multi_word_pattern[i + 8] = pattern_32b[i];
		}
		wr_pattern = multi_word_pattern[0]; // for printf
	}
	else { wr_pattern = idd4r; shift_pattern = true; use_multi_word_pattern = false; }

	// Select the appropriate power measurement pattern based on access_pattern
	if(access_pattern == 0) {
		program_idd0(ch, pc, bg0, bg1);
	} else if(access_pattern == 1) {
		program_idd4w(ch, pc, bg0, bg1);
	} else if(access_pattern == 2) {
		program_idd2(ch, pc);
	} else if(access_pattern == 3) {
		program_idd3n1(ch, pc);
	} else if(access_pattern == 18) {
		program_idd3n16(ch, pc);
	} else if(access_pattern == 5) {
		program_idd5(ch, pc, bg0, bg1);
	} else if(access_pattern == 9) {
		program_idd4r_bitflip(ch, pc, bg0, bg1);
	} else if(access_pattern == 10) {
		program_idd4r_custom_data(ch, pc, bg0, bg1);
	} else if(access_pattern == 11) {
		program_max_power_loop(ch, pc, bg0, bg1);
	} else if(access_pattern == 12) {
		program_idd1_precharge_all(ch, pc);
	} else if(access_pattern == 13) {
		program_idd1_pipelined(ch, pc, bg0, bg1);
	} else if(access_pattern == 14) {
		program_idd1_pipelined_bankoffset(ch, pc, bg0, bg1);
	} else if(access_pattern == 15) {
		program_idd1_rdap(ch, pc, bg0, bg1);
	} else if(access_pattern == 16) {
		program_idd1_rdap_bankoffset(ch, pc, bg0, bg1);
	} else if(access_pattern == 17) {
		program_idd1_max_actpre(ch, pc, bg0, bg1);
	} else if(access_pattern == 19) {
		program_idd7(ch, pc, bg0, bg1);
	} else if(access_pattern == 21) {
		program_idd7_experimental(ch, pc, bg0, bg1);
	} else if(access_pattern == 22) {
		program_max_power_16bank(ch, pc, bg0, bg1);
	} else if(access_pattern == 20) {
		program_idd1_rdap_singlepass(ch, pc, bg0, bg1);
	} else if(access_pattern == 23) {
		program_idd7_noread(ch, pc, bg0, bg1);
	} else if(access_pattern == 24) {
		program_idd7_actpre_maxpower(ch, pc, bg0, bg1);
	} else if(access_pattern == 25) {
		program_idd7_actpre_maxpower_half(ch, pc, bg0, bg1);
	} else if(access_pattern == 26) {
		program_idd7_maxpower_rdonly(ch, pc, bg0, bg1);
	} else if(access_pattern == 27) {
		program_max_power_loop_slot13(ch, pc, bg0, bg1);
	} else if(access_pattern == 28) {
		program_max_power_loop_3col(ch, pc, bg0, bg1);
	} else if(access_pattern == 29) {
		program_idd4r_4bank_2bg(ch, pc, bg0, bg1);
	} else if(access_pattern == 30) {
		program_idd4r_2bank(ch, pc, bg0, bg1);
	} else if(access_pattern == 31) {
		program_idd4r_2bank_sparse(ch, pc, bg0, bg1);
	} else if(access_pattern == 32) {
		program_idd4r_8bank_4bg(ch, pc, bg0, bg1);
	} else if(access_pattern == 33) {
		program_idd4r_4bank_1bg(ch, pc, bg0, bg1);
	} else if(access_pattern == 34) {
		program_idd4r_3bank_3bg(ch, pc, bg0);
	} else if(access_pattern == 35) {
		program_idd4r_full_32bank(ch, pc, bg0, bg1);
	} else {
		// For idd4r patterns (default)
		if(use_bank_variation)
			program_idd4r_bank_variation(ch, pc);
		else if(use_full_idd4r)
			program_idd4r_full(ch, pc, bg0, bg1);
		else
			program_idd4r(ch, pc, bg0, bg1);
	}
	// Initialize the platform, opens file descriptors for the board PCI-E interface.
	// reset the board to hopefully restore the board's state
	platform->reset_fpga();

	platform->set_garbage_reads(GARBAGE_READS ? true : false);

	platform->initializeMonitoring();

	if(MULTI_CHANNEL) {
		platform->set_broadcast_channels(channels_to_broadcast);

		std::cout << "Channels to broadcast to:";
		for(auto &_ch : channels_to_broadcast) std::cout << " " << _ch;
		std::cout << std::endl;
	}

	// Optional whole-DRAM pre-fill (both PCs of the broadcast channels) BEFORE the
	// metrics thread starts, so the fill itself is not measured. The filled data
	// is the controlled background for resident-data-dependent IDD tests.
	if(fill_whole_mode != 0) {
		std::cout << "Pre-filling whole DRAM (mode="
			<< (fill_whole_mode == 1 ? "zeros" : "random")
			<< ", seed=0x" << std::hex << fill_seed << std::dec << ")..." << std::endl;
		run_whole_fill(platform, ch, fill_whole_mode, fill_seed);
		std::cout << "Pre-fill complete." << std::endl;
	}

	/* SoftMC programs are formed sequentially by adding
	 * instructions to the program one by one. Instructions
	 * can be added to the program using add_inst().
	 */


	// Transfer the program to the FPGA board
	
	if(NUM_REPS < 10) {
		program.pretty_print();
	}

	memset(buf, 0, sizeof(buf));
	// platform.reset_fpga();
	// platform.initializeMonitoring();
	if(GARBAGE_READS) {
		// If doing garbage reads, we want to start the metrics thread before executing the program, to capture the power stabilization phase
		if(fixed_duration_s > 0)
			platform->startMetricsThreadFixedDuration(1000, fixed_duration_s);
		else
			platform->startMetricsThread(1000); // Record metrics every 1 s
	}
	platform->resetMonitoring();
	platform->execute(program);
	// long long size = (INNER_REPS * (long long) NUM_REPS * 8 * NUM_PC * (long long) BYTES_PER_READ); // Read 16 cache lines (32 bytes each) from both PCs
	long long size = std::max((2 * NUM_PC * (long long) BYTES_PER_READ), (long long)BUFFER_SIZE); // Read 16 cache lines (32 bytes each) from both PCs

	if(!GARBAGE_READS) {
		for(long long c = 0; c < size; c += BUFFER_SIZE) {
			platform->receiveData(buf, BUFFER_SIZE);
			for(int i = 0; i < BUFFER_SIZE; i++) {
				printf("buf[%d]=%x\n", i, buf[i]);
			}
			
		}
		// platform->stopMetricsThread();
		platform->readAndPrintHBMMetrics();
		// sleep(1);
	}else {
		// Wait for metrics thread to stop (which indicates power has stabilized to below max)
		int metrics_wait = 0;
		while (platform->metricsThreadRunning())
		{
			sleep(1);
			metrics_wait++;
			std::cout << "Waiting for metrics thread to stop (power stabilization)... " << metrics_wait << "s" << std::endl;
		}
		std::cout << "Program ending after metrics thread stopped (power stabilized)" << std::endl;
		platform->stopMetricsThread();
	}

	// IDD2 cooldown: run idle pattern for 10 seconds to contain HBM temperature
	// std::cout << "Starting IDD2 cooldown (10 seconds)..." << std::endl;
	platform->reset_fpga();

	// Program cooldown_prog;
	// program_idd2_cooldown(cooldown_prog, ch, pc);

	// if(MULTI_CHANNEL) {
	// 	platform->set_broadcast_channels(channels_to_broadcast);
	// }

	// platform->execute(cooldown_prog);
	// sleep(10);
	// platform->reset_fpga();
	// std::cout << "IDD2 cooldown complete." << std::endl;
}

int main(int argc , char *argv[])
{
	if (argc < 2)
	{
		printf("Usage: ./SoftMC_rdwr test_select [access_pattern] [pattern] [num_excluded] [excluded_channels...]\n");
		printf("Optional flags (after positional args):\n");
		printf("  --pc PC                     Pseudo channel (0 or 1, default: 0)\n");
		printf("  --bg BG0 BG1                Bank group pair (0-3, default: 0 1)\n");
		printf("  --banks B0 B1 B2 B3         Bank offsets within groups (0-3, default: 0 1 2 3)\n");
		printf("  --bank-variation            Bank variation mode: 4 banks, one per BG (0+a, 4+b, 8+c, 12+d)\n");
		printf("  --rows ROW0 ROW1            Row addresses (hex ok, default: 0x5555 0x2aaa)\n");
		printf("  --cols COL0 COL1            Column addresses (hex ok, default: 10 21)\n");
		printf("  --use-full                  Use idd4r_full pattern (all 4 bank groups)\n");
		printf("  --fixed-duration [SECONDS]  Skip stabilization; record for a fixed duration (default: 3600 = 1h)\n");
		printf("  --freq MHZ                  Frequency in MHz (appended to CSV filename)\n");
		printf("  --bitflip MASK              8-bit flip mask for bitflip test (0-255, default: 0)\n");
		printf("  --pattern-32b W0..W7        32-byte pattern as 8 hex uint32_t values\n");
		printf("  --invert-col1               Invert data pattern for col1 addresses (for access_pattern 10)\n");
		printf("  --no-garbage-reads          Disable garbage reads (print received data instead)\n");
		printf("  --fill-whole zeros|random   Pre-fill the WHOLE DRAM (both PCs, all banks/rows/cols)\n");
		printf("                              before the measurement (zeros, or per-row random data)\n");
		printf("  --fill-seed N               Seed for --fill-whole random (default: 0xBADC0FFE)\n");
		exit(0);
	}

	char *testselect = argv[1];
	int test_select = atoi(testselect);
	access_pattern = (argc > 2) ? atoi(argv[2]) : 2;
	pattern = (argc > 3) ? atoi(argv[3]) : 2;
	int num_channels = (argc > 4) ? atoi(argv[4]) : 1;
	assert(num_channels >= 0 && num_channels < 16);
	int flags_start = 5 + num_channels;

	std::set<int> to_exclude;
	for(int i = 0; i < num_channels && (5 + i) < argc; i++) {
		to_exclude.insert(atoi(argv[5 + i]));
	}

	for(int i = 0; i < 16; i++) {
		if(to_exclude.find(i) == to_exclude.end()) channels_to_broadcast.push_back(i);
	}

	int pc = 0;
	int bg0 = 0;
	int bg1 = 1;

	fixed_duration_s = 50;

	// Parse optional flags
	for(int i = flags_start; i < argc; i++) {
		if(strcmp(argv[i], "--pc") == 0 && i + 1 < argc) {
			pc = atoi(argv[++i]);
		} else if(strcmp(argv[i], "--bg") == 0 && i + 2 < argc) {
			bg0 = atoi(argv[++i]);
			bg1 = atoi(argv[++i]);
		} else if(strcmp(argv[i], "--banks") == 0 && i + 4 < argc) {
			for(int j = 0; j < 4; j++) bank_offsets[j] = atoi(argv[++i]);
		} else if(strcmp(argv[i], "--rows") == 0 && i + 2 < argc) {
			row_addr0 = (uint16_t)strtol(argv[++i], NULL, 0);
			row_addr1 = (uint16_t)strtol(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--cols") == 0 && i + 2 < argc) {
			col_addr0 = (uint16_t)strtol(argv[++i], NULL, 0);
			col_addr1 = (uint16_t)strtol(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--use-full") == 0) {
			use_full_idd4r = true;
		} else if(strcmp(argv[i], "--bank-variation") == 0) {
			use_bank_variation = true;
		} else if(strcmp(argv[i], "--custom-pattern") == 0 && i + 1 < argc) {
			custom_wr_pattern = (uint32_t)strtoul(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--fixed-duration") == 0) {
			fixed_duration_s = atoi(argv[++i]);
		} else if(strcmp(argv[i], "--freq") == 0 && i + 1 < argc) {
			freq_mhz = atoi(argv[++i]);
		} else if(strcmp(argv[i], "--bitflip") == 0 && i + 1 < argc) {
			bitflip_mask = (uint8_t)strtoul(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--pattern-32b") == 0 && i + 8 < argc) {
			for(int j = 0; j < 8; j++) pattern_32b[j] = (uint32_t)strtoul(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--pattern-32b-B") == 0 && i + 8 < argc) {
			for(int j = 0; j < 8; j++) pattern_32b_B[j] = (uint32_t)strtoul(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--pattern-32b-C") == 0 && i + 8 < argc) {
			for(int j = 0; j < 8; j++) pattern_32b_C[j] = (uint32_t)strtoul(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--col2") == 0 && i + 1 < argc) {
			col_addr2 = (uint16_t)strtol(argv[++i], NULL, 0);
		} else if(strcmp(argv[i], "--invert-col1") == 0) {
			invert_col1 = true;
		} else if(strcmp(argv[i], "--no-garbage-reads") == 0) {
			garbage_reads_enabled = 0;
		} else if(strcmp(argv[i], "--fill-whole") == 0 && i + 1 < argc) {
			const char *m = argv[++i];
			if(strcmp(m, "zeros") == 0) fill_whole_mode = 1;
			else if(strcmp(m, "random") == 0) fill_whole_mode = 2;
			else { fprintf(stderr, "--fill-whole must be 'zeros' or 'random'\n"); exit(1); }
		} else if(strcmp(argv[i], "--fill-seed") == 0 && i + 1 < argc) {
			fill_seed = (uint32_t)strtoul(argv[++i], NULL, 0);
		}
	}

	int ch = channels_to_broadcast[0];

	printf("Configuration:\n");
	printf("  Channels: ");
	for(auto &c : channels_to_broadcast) printf("%d ", c);
	printf("\n");
	printf("  PC: %d, BG pair: (%d, %d)\n", pc, bg0, bg1);
	printf("  Bank offsets: [%d, %d, %d, %d]\n", bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3]);
	if(use_bank_variation)
		printf("  Bank variation mode: banks [%d, %d, %d, %d]\n",
			0 + bank_offsets[0], 4 + bank_offsets[1], 8 + bank_offsets[2], 12 + bank_offsets[3]);
	printf("  Row addresses: 0x%04x, 0x%04x\n", row_addr0, row_addr1);
	printf("  Column addresses: 0x%02x, 0x%02x\n", col_addr0, col_addr1);
	printf("  Use full idd4r: %s\n", use_full_idd4r ? "yes" : "no");
	printf("  Bank variation: %s\n", use_bank_variation ? "yes" : "no");
	if(pattern == 4 || pattern == 5)
		printf("  Custom pattern: 0x%08x (%s shifting)\n", custom_wr_pattern, pattern == 5 ? "with" : "no");
	if(pattern == 6)
		printf("  Multi-word pattern: 00-00-55-55-FF-FF-AA-AA (x2)\n");
	if(pattern == 7) {
		printf("  Pattern 32b:");
		for(int i = 0; i < 8; i++) printf(" 0x%08x", pattern_32b[i]);
		printf("\n");
	}
	if(access_pattern == 9)
		printf("  Bitflip mask: 0x%02x\n", bitflip_mask);
	if(access_pattern == 10)
		printf("  Invert col1: %s\n", invert_col1 ? "yes" : "no");
	if(access_pattern == 28) {
		printf("  3-col mode: col2=0x%02x\n", col_addr2);
		printf("  Pattern B:");
		for(int i = 0; i < 8; i++) printf(" 0x%08x", pattern_32b_B[i]);
		printf("\n  Pattern C:");
		for(int i = 0; i < 8; i++) printf(" 0x%08x", pattern_32b_C[i]);
		printf("\n");
	}
	printf("  Fixed-duration mode: %s\n", fixed_duration_s > 0 ? (std::to_string(fixed_duration_s) + "s").c_str() : "off (stabilization)");
	if(freq_mhz > 0)
		printf("  Frequency tag: %d MHz\n", freq_mhz);

	SoftMCPlatform platform(0, GARBAGE_READS);
	g_platform = &platform;
	std::signal(SIGINT, sigint_handler);

	std::string csv_path, ch_string, pat_string, access_pat_string;
	ch_string = "";
	for(const auto u : channels_to_broadcast) ch_string += "_" + std::to_string(u);

	if(ACCESS_PATTERN == 1) access_pat_string = "idd4w";
	else if(ACCESS_PATTERN == 2) access_pat_string = "idd2";
	else if(ACCESS_PATTERN == 3) access_pat_string = "idd3n1";
	else if(ACCESS_PATTERN == 18) access_pat_string = "idd3n16";
	else if(ACCESS_PATTERN == 0) access_pat_string = "idd0";
	else if(ACCESS_PATTERN == 5) access_pat_string = "idd5";
	else if(ACCESS_PATTERN == 7) access_pat_string = "idd7";
	else if(ACCESS_PATTERN == 6) access_pat_string = "idd7_hbm3";
	else if(ACCESS_PATTERN == 4) access_pat_string = use_full_idd4r ? "idd4r_16bank" : "idd4r_full";
	else if(ACCESS_PATTERN == 8) access_pat_string = "activates";
	else if(ACCESS_PATTERN == 9) access_pat_string = "idd4r_bitflip";
	else if(ACCESS_PATTERN == 10) access_pat_string = "idd4r_custom_data";
	else if(ACCESS_PATTERN == 11) access_pat_string = "max_power_loop";
	else if(ACCESS_PATTERN == 12) access_pat_string = "idd1_precharge_all";
	else if(ACCESS_PATTERN == 13) access_pat_string = "idd1_pipelined";
	else if(ACCESS_PATTERN == 14) access_pat_string = "idd1_pipelined_bankoffset";
	else if(ACCESS_PATTERN == 15) access_pat_string = "idd1_rdap";
	else if(ACCESS_PATTERN == 16) access_pat_string = "idd1_rdap_bankoffset";
	else if(ACCESS_PATTERN == 17) access_pat_string = "idd1_max_actpre";
	else if(ACCESS_PATTERN == 18) access_pat_string = "idd3n16";
	else if(ACCESS_PATTERN == 19) access_pat_string = "idd7";
	else if(ACCESS_PATTERN == 21) access_pat_string = "idd7_experimental";
	else if(ACCESS_PATTERN == 22) access_pat_string = "max_power_16bank";
	else if(ACCESS_PATTERN == 20) access_pat_string = "idd1_rdap_singlepass";
	else if(ACCESS_PATTERN == 23) access_pat_string = "idd7_noread";
	else if(ACCESS_PATTERN == 24) access_pat_string = "idd7_actpre_maxpower";
	else if(ACCESS_PATTERN == 25) access_pat_string = "idd7_actpre_maxpower_half";
	else if(ACCESS_PATTERN == 26) access_pat_string = "idd7_maxpower_rdonly";
	else if(ACCESS_PATTERN == 27) access_pat_string = "max_power_loop_slot13";
	else if(ACCESS_PATTERN == 28) access_pat_string = "max_power_loop_3col";
	else if(ACCESS_PATTERN == 29) access_pat_string = "idd4r_4bank_2bg";
	else if(ACCESS_PATTERN == 30) access_pat_string = "idd4r_2bank";
	else if(ACCESS_PATTERN == 31) access_pat_string = "idd4r_2bank_sparse";
	else if(ACCESS_PATTERN == 32) access_pat_string = "idd4r_8bank_4bg";
	else if(ACCESS_PATTERN == 33) access_pat_string = "idd4r_4bank_1bg";
	else if(ACCESS_PATTERN == 34) access_pat_string = "idd4r_3bank_3bg";
	else if(ACCESS_PATTERN == 35) access_pat_string = "idd4r_full_32bank";
	else access_pat_string = "idd4r";

	if(use_full_idd4r && access_pat_string == "idd4r") access_pat_string = "idd4r_full";

	if(PATTERN == 0) pat_string = "zeros";
	else if(PATTERN == 1) pat_string = "ones";
	else if(PATTERN == 2) pat_string = "0055ffaa";
	else if (PATTERN == 3) pat_string = "aaaaaaaa";
	else if (PATTERN == 4 || PATTERN == 5) {
		char pat_buf[64];
		snprintf(pat_buf, sizeof(pat_buf), "custom_%08x%s", custom_wr_pattern, PATTERN == 5 ? "_shift" : "_noshift");
		pat_string = pat_buf;
	}
	else if (PATTERN == 6) pat_string = "multi_00_55_ff_aa";
	else if (PATTERN == 7) {
		if(access_pattern == 28) {
			// Compact encoding: 4-bit beat code per pattern (A, B, C)
			int codeA = 0, codeB = 0, codeC = 0;
			for(int beat = 0; beat < 4; beat++) {
				if(pattern_32b[beat*2] == 0xFFFFFFFF) codeA |= (1 << beat);
				if(pattern_32b_B[beat*2] == 0xFFFFFFFF) codeB |= (1 << beat);
				if(pattern_32b_C[beat*2] == 0xFFFFFFFF) codeC |= (1 << beat);
			}
			char pat_buf[64];
			snprintf(pat_buf, sizeof(pat_buf), "3col_A%x_B%x_C%x", codeA, codeB, codeC);
			pat_string = pat_buf;
		} else {
			char pat_buf[128];
			snprintf(pat_buf, sizeof(pat_buf), "32b_%08x_%08x_%08x_%08x_%08x_%08x_%08x_%08x",
				pattern_32b[0], pattern_32b[1], pattern_32b[2], pattern_32b[3],
				pattern_32b[4], pattern_32b[5], pattern_32b[6], pattern_32b[7]);
			pat_string = pat_buf;
		}
	}
	else pat_string = "0055ffaa";

	// Map test_select to output directory name
	std::string output_dir;
	switch(test_select) {
		case 1: output_dir = "results/channel_variation_hypothesis_temperature_stable"; break;
		case 2: output_dir = "results/pseudo_channel_variation"; break;
		case 3: output_dir = "results/bank_group_variation"; break;
		case 4: output_dir = "results/bank_variation"; break;
		case 5: output_dir = "results/row_variation_new"; break;
		case 6: output_dir = "results/column_variation"; break;
		case 7: output_dir = "results/temperature_dependence"; break;
		case 8: output_dir = "results/row_pattern_variation_new"; break;
		case 9: output_dir = "results/data_pattern_variation"; break;
		case 10: output_dir = "results/bitflip_variation"; break;
		case 11: output_dir = "results/custom_data_variation"; break;
		case 12: output_dir = "results/max_power_channel_combo"; break;
		case 13: output_dir = "results/max_power_bank_group"; break;
		case 14: output_dir = "results/max_power_bank_offset"; break;
		case 15: output_dir = "results/max_power_bg_invert"; break;
		case 16: output_dir = "results/pc_variation_test"; break;
		case 17: output_dir = "results/frequency_variation"; break;
		case 18: output_dir = "results/idd1_variation"; break;
		case 19: output_dir = "results/idd1_2a_row_variation"; break;
		case 20: output_dir = "results/idd1_3b_row_variation"; break;
		case 21: output_dir = "results/idd1_4_row_variation"; break;
		case 22: output_dir = "results/idd7_experimentation"; break;
		case 23: output_dir = "results/max_power_16bank"; break;
		case 24: output_dir = "results/beat_pattern_variation"; break;
		case 25: output_dir = "results/data_pattern_variation_new"; break;
		case 26: output_dir = "results/data_pattern_variation_bank_subset"; break;
		case 27: output_dir = "results/idd4r_maxpower_combined"; break;
		case 28: output_dir = "results/max_power_bg_invert_sid"; break;
		case 29: output_dir = "results/idd4r_4bank_2bg_sid"; break;
		case 30: output_dir = "results/max_power_channel_combo_bg46"; break;
		case 31: output_dir = "results/max_power_channel_pairs_bg46"; break;
		case 32: output_dir = "results/idd4r_random_patterns"; break;
		case 33: output_dir = "results/idd4w_random_patterns"; break;
		case 34: output_dir = "results/idd_prefill"; break;
		default: output_dir = "results/test_" + std::to_string(test_select); break;
	}
	
	mkdir("results", 0777); // Create results directory if it doesn't exist
	if (fixed_duration_s > 0) {
		output_dir += "_fixed_reset";
	}
	output_dir += "_full_ipp";
	mkdir(output_dir.c_str(), 0755);

	// Build CSV filename with all parameters
	char extra_buf[256];
	if(access_pattern == 28) {
		snprintf(extra_buf, sizeof(extra_buf), "_pc%d_bg%d_%d_banks%d%d%d%d_rows%04x_%04x_cols%02x_%02x_%02x",
			pc, bg0, bg1,
			bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3],
			row_addr0, row_addr1, col_addr0, col_addr1, col_addr2);
	} else {
		snprintf(extra_buf, sizeof(extra_buf), "_pc%d_bg%d_%d_banks%d%d%d%d_rows%04x_%04x_cols%02x_%02x",
			pc, bg0, bg1,
			bank_offsets[0], bank_offsets[1], bank_offsets[2], bank_offsets[3],
			row_addr0, row_addr1, col_addr0, col_addr1);
	}

	std::string bitflip_tag;
	if(access_pattern == 9) {
		char bf_buf[8];
		snprintf(bf_buf, sizeof(bf_buf), "_bf%02x", bitflip_mask);
		bitflip_tag = bf_buf;
	}
	std::string invert_tag;
	if(access_pattern == 10 || access_pattern == 11) {
		invert_tag = invert_col1 ? "_inv1" : "_inv0";
	}
	std::string duration_tag = (fixed_duration_s > 0) ? "_dur" + std::to_string(fixed_duration_s) + "s" : "";
	std::string freq_tag = (freq_mhz > 0) ? "_freq" + std::to_string(freq_mhz) + "MHz" : "";
	std::string fill_tag;
	if(fill_whole_mode == 1) fill_tag = "_fillzeros";
	else if(fill_whole_mode == 2) {
		char fb[32];
		snprintf(fb, sizeof(fb), "_fillrand_s%08x", fill_seed);
		fill_tag = fb;
	}
	csv_path = output_dir + "/hbm_" + access_pat_string + "_" + pat_string + ch_string + std::string(extra_buf) + bitflip_tag + invert_tag + fill_tag + duration_tag + freq_tag + ".csv";

	platform.setCSVFilePath(csv_path);

	int err;
	if((err = platform.init()) != SOFTMC_SUCCESS){
		cerr << "Could not initialize SoftMC Platform: " << err << endl;
	}

	power_measurement(&platform, test_select, ch, pc, bg0, bg1);
	return 0;
}
