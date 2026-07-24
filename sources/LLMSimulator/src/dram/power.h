#pragma once

#include <type_traits>

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

struct DramEnergy {
  double kACT_energy_j_ = 0.0;
  double kREAD_energy_j_ = 0.0;
  double kWRITE_energy_j_ = 0.0;
  double kALL_ACT_energy_j_ = 0.0;
  double kALL_READ_energy_j_ = 0.0;
  double kALL_WRITE_energy_j_ = 0.0;
  double kMAC_energy_j_ = 0.0;
};

// 2017 MICRO FGDRAM
// https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
// ACT energy = 0.909 nJ
// Cell (RD/WRT) energy: 0.44pJ/b,
// RD/WR Energy (column decoder to BG MUX): 1.01 pJ/b
// RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/b
// TSV energy : 0.5 pJ/b
// Silicon interposer IO energy : 0.3 pJ/b
// DRAM cell to power TSV area + TSV (w/ TSV energy) = 1.81 pJ/b

// energy per bit of Read(RD) and Write(WR) assumed to be the same
// we multiply this value to the number of count (not number of bits)
// for example, energy for Read operation in HBM2E is 3.48 (= 0.44 + 1.01 + 1.23 + 0.5 + 0.3) pJ/b
// and HBM's granularity is 32Byte (256bit), so we multiply 256 and divide it by 1000 to get nJ (= 3.48 * 256 / 1000 = 0.8912 nJ)

static DramEnergy gpuEnergy{0.909, 0.891, 0.891, 0, 0, 0, 0.46 / 2 / 1000};

static DramEnergy logicEnergy{0.909,     0.464,     0.464,
                              0.909 * 8, 0.464 * 8, 0.464 * 8, 0.46 / 2 / 1000};  // X4

static DramEnergy pimBankgroupEnergy{0.909,     0.686,     0.686,
                                     0.909 * 8, 0.686 * 8, 0.686 * 8, 0.46 / 2 / 1000};  // X4

static DramEnergy pimEnergy{0.909,      0.187,      0.187,
                            0.909 * 32, 0.187 * 32, 0.187 * 32, 0.46 / 2 / 1000};  // X16

}  // namespace llm_system
