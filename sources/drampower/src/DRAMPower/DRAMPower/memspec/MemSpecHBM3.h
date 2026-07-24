#ifndef DRAMPOWER_MEMSPEC_MEMSPECHBM3_H
#define DRAMPOWER_MEMSPEC_MEMSPECHBM3_H

// First-cut HBM3 memory spec for the DRAMPower engine. Deliberately mirrors the
// self-contained MemSpecHBM2 (same VDD-only power model + per-bankgroup/bank IDD
// variation) so the HBM3 core-energy math can be a direct port of the HBM2 one.
// HBM3 differs from HBM2 only parametrically here (e.g. 16 channels x 2 pseudo-
// channels, 32-bit pseudo-channel, BL16); those are supplied via the config JSON,
// not hard-coded. HBM3-specific timings (tFAW, tRRD, per-bank refresh, refresh
// management) can be added in a later step if finer modeling is needed.

#include <DRAMPower/command/CmdType.h>
#include <DRAMPower/util/datapattern_model.h>

#include <cstdint>
#include <string>
#include <vector>

namespace DRAMPower {

class MemSpecHBM3
{
public:
    enum VoltageDomain {
        VDD = 0,
    };

    struct MemArchitectureSpec
    {
        uint64_t nbrOfChannels;
        uint64_t nbrOfPseudoChannels;
        uint64_t nbrOfBankGroups;
        uint64_t nbrOfBanks;       // banks per bank group
        uint64_t nbrOfRows;
        uint64_t nbrOfColumns;
        uint64_t burstLength;
        uint64_t dataRate;
        uint64_t width;            // device bit width
    };

    struct MemTimingSpec
    {
        double tCK;                // clock period in ps
        uint64_t tRAS;
        uint64_t tRCD;
        uint64_t tRCDWR;
        uint64_t tRC;
        uint64_t tRL;              // CAS latency (nCL)
        uint64_t tWL;
        uint64_t tCCD_S;
        uint64_t tCCD_L;
        uint64_t tWTR_S;
        uint64_t tWTR_L;
        uint64_t tWR;
        uint64_t tRP;
        uint64_t tBurst;
    };

    struct MemPowerSpec
    {
        double vDD;
        double vDDQ;               // I/O supply (HBM4 split rail); applied to the DQ
                                   // component of the read I/O current only. Defaults
                                   // to vDD, recovering the single-rail behaviour.
        double iDD0;
        double iDD2N;
        double iDD3N1;             // active standby, 1 bank open
        double iDD3N16;            // active standby, all banks open
        double iDD4R;
        double iDD4W;

        std::vector<double> bankgroup_scaling_factors; // to account for variation between bankgroups
        std::vector<double> bank_scaling_factors; // to account for variation between banks
    };

    struct BankWiseParams
    {
        double bwPowerFactRho;
    };

public:
    MemSpecHBM3() = delete;

    MemSpecHBM3(const MemArchitectureSpec &archSpec,
                const MemTimingSpec &timingSpec,
                const MemPowerSpec &powerSpec);

    MemSpecHBM3(const MemSpecHBM3&) = default;
    MemSpecHBM3(MemSpecHBM3&&) = default;
    MemSpecHBM3& operator=(MemSpecHBM3&&) = default;
    ~MemSpecHBM3() = default;

    uint64_t timeToCompletion(CmdType type);

    // Architecture
    uint64_t numberOfChannels;
    uint64_t numberOfPseudoChannels;
    uint64_t numberOfBankGroups;
    uint64_t numberOfBanks;            // per bank group
    uint64_t numberOfRows;
    uint64_t numberOfColumns;
    uint64_t burstLength;
    uint64_t dataRate;
    uint64_t bitWidth;

    uint64_t banksPerPseudoChannel;    // bankGroups * banksPerGroup

    uint64_t prechargeOffsetRD;
    uint64_t prechargeOffsetWR;

    MemTimingSpec memTimingSpec;
    MemPowerSpec memPowerSpec;
    BankWiseParams bwParams;

    // Optional data-pattern read/write energy model (disabled by default).
    util::DataPatternModel dataPattern;
};

} // namespace DRAMPower

#endif /* DRAMPOWER_MEMSPEC_MEMSPECHBM3_H */
