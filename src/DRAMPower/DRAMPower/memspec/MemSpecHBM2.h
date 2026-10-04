#ifndef DRAMPOWER_MEMSPEC_MEMSPECHBM2_H
#define DRAMPOWER_MEMSPEC_MEMSPECHBM2_H

#include <DRAMPower/command/CmdType.h>
#include <DRAMPower/util/datapattern_model.h>

#include <cstdint>
#include <string>
#include <vector>

namespace DRAMPower {

class MemSpecHBM2
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
        uint64_t tRFC;             // all-bank refresh cycle time (cycles); 0 = refresh energy not modeled
        uint64_t tBurst;
    };

    struct MemPowerSpec
    {
        double vDD;
        double iDD0;
        double iDD2N;
        double iDD3N1;             // active standby, 1 bank open
        double iDD3N16;            // active standby, all banks open
        double iDD4R;
        double iDD4W;
        double iDD5B = 0.0;        // all-bank refresh burst current; 0 = refresh energy not modeled

        std::vector<double> bankgroup_scaling_factors; // to account for variation between bankgroups
        std::vector<double> bank_scaling_factors; // to account for variation between banks
        // Optional write-specific factors; an empty vector falls back to the read factors above.
        std::vector<double> bankgroup_write_scaling_factors;
        std::vector<double> bank_write_scaling_factors;
    };

    struct BankWiseParams
    {
        double bwPowerFactRho;
    };

public:
    MemSpecHBM2() = delete;

    MemSpecHBM2(const MemArchitectureSpec &archSpec,
                const MemTimingSpec &timingSpec,
                const MemPowerSpec &powerSpec);

    MemSpecHBM2(const MemSpecHBM2&) = default;
    MemSpecHBM2(MemSpecHBM2&&) = default;
    MemSpecHBM2& operator=(MemSpecHBM2&&) = default;
    ~MemSpecHBM2() = default;

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

#endif /* DRAMPOWER_MEMSPEC_MEMSPECHBM2_H */
