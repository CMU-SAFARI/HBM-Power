#pragma once
#include "common/type.h"

namespace llm_system {

enum class MMap {
  ALL_CHANNEL = 0,
  CUBE_CHANNEL,
  CHANNEL,
};

enum class DRAMRequestType {
  kRead = 0,   // None-pim
  kWrite = 1,  // None-pim
  kMove = 2,
  kMult = 3,
  kAdd = 4,
  kMAD = 5,
  kPMult = 6,
  kCMult = 7,
  kCAdd = 8,
  kCMAD = 9,
  kTensor = 10,
  kTensor_Square = 11,
  kModup_Evkmult = 12,
  kModDownEpilogue = 13,
  kPMult_Accum = 14,
  kCMult_Accum = 15,
  kGEMV = 16,
  kMAX,
};

enum class PIMCommandType {
  kAdd = 0,  // r0 = r1 + r2
  kSub,      // r0 = r1 - r2
  kMult,     // r0 = r1 * r2
  kMAC,      // r0 = r1 * r2 + r3
  kDRAM2RF,  // Move, DRAM to register file
  kRF2DRAM,  // Move, register file to DRAM
  kRead,     // Generic read
  kWrite,    // Generic Write
  kMAX,
};

enum class DRAMCommandType {
  kACT = 0,
  kPRE,
  kPREA,
  kALL_PRE,
  kREAD,
  kWRITE,
  kALL_ACT,
  kALL_READ,
  kALL_WRITE,
  kREF,
  kMAX,
};

enum class LPDDR5CommandType { // LPDDR5
  kACT_1 = 0,
  kACT_2,
  kALL_ACT_1,
  kALL_ACT_2,
  kPRE,
  kPREA,
  kALL_PRE,
  kCASRD,
  kCASWR,
  kRD16,
  kWR16,
  kRD16A,
  kWR16A,
  kALL_RD,
  kALL_WR,
  kREF,
  kMAX,
};

enum class PIMOperandType {
  kRF = 0,
  kDRAM = 1,
  kSrc = 2,
  kPrecomputed = 3,
  kDest = 4,
  kModUp = 5,  // deprecated
  kEvk = 6,    // deprecated
  kMAX = 7,
};

}  // namespace llm_system
