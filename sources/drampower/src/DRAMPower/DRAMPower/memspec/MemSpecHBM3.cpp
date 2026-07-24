#include "MemSpecHBM3.h"

using namespace DRAMPower;

MemSpecHBM3::MemSpecHBM3(const MemArchitectureSpec &archSpec,
                         const MemTimingSpec &timingSpec,
                         const MemPowerSpec &powerSpec)
{
    // Architecture
    numberOfChannels        = archSpec.nbrOfChannels;
    numberOfPseudoChannels  = archSpec.nbrOfPseudoChannels;
    numberOfBankGroups      = archSpec.nbrOfBankGroups;
    numberOfBanks           = archSpec.nbrOfBanks;
    numberOfRows            = archSpec.nbrOfRows;
    numberOfColumns         = archSpec.nbrOfColumns;
    burstLength             = archSpec.burstLength;
    dataRate                = archSpec.dataRate;
    bitWidth                = archSpec.width;

    banksPerPseudoChannel   = numberOfBankGroups * numberOfBanks;

    // Timing
    memTimingSpec = timingSpec;
    memTimingSpec.tBurst = burstLength / dataRate;

    // Power (VDD domain only)
    memPowerSpec = powerSpec;

    // Bank-wise defaults
    bwParams.bwPowerFactRho = 1.0;

    // Precharge offsets
    prechargeOffsetRD = memTimingSpec.tRL + memTimingSpec.tBurst;
    prechargeOffsetWR = memTimingSpec.tBurst + memTimingSpec.tWL + memTimingSpec.tWR;
}

uint64_t MemSpecHBM3::timeToCompletion(DRAMPower::CmdType type)
{
    uint64_t offset = 0;

    if (type == CmdType::ACT)
        offset = memTimingSpec.tRCD;
    else if (type == CmdType::RD)
        offset = memTimingSpec.tRL + memTimingSpec.tBurst;
    else if (type == CmdType::WR)
        offset = memTimingSpec.tWL + memTimingSpec.tBurst;
    else if (type == CmdType::PRE || type == CmdType::PREA)
        offset = memTimingSpec.tRP;

    return offset;
}
