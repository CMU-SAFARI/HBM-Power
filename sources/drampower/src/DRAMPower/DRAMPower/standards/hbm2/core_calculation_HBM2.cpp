#include "core_calculation_HBM2.h"

#include <DRAMPower/standards/hbm2/HBM2.h>

#include <cassert>

namespace DRAMPower {

    energy_t Calculation_HBM2::calcEnergy(timestamp_t timestamp, HBM2 &dram) {
        auto stats = dram.getWindowStats(timestamp);

        double t_CK = dram.memSpec.memTimingSpec.tCK;
        auto t_RAS = dram.memSpec.memTimingSpec.tRAS * t_CK;
        auto t_RP = dram.memSpec.memTimingSpec.tRP * t_CK;

        auto rho = dram.memSpec.bwParams.bwPowerFactRho;
        auto BL = dram.memSpec.burstLength;
        auto DR = dram.memSpec.dataRate;
        auto B = dram.memSpec.banksPerPseudoChannel;

        auto VDD = dram.memSpec.memPowerSpec.vDD;            // V
        // IDD values are specified in mA; convert to A so that
        // V * A * ps = pJ (picojoules)
        auto IDD0    = dram.memSpec.memPowerSpec.iDD0    * 1e-3;
        auto IDD2N   = dram.memSpec.memPowerSpec.iDD2N   * 1e-3;
        auto IDD3N1  = dram.memSpec.memPowerSpec.iDD3N1  * 1e-3;
        auto IDD3N16 = dram.memSpec.memPowerSpec.iDD3N16 * 1e-3;
        // Read/write currents: optionally replaced by the data-pattern model, which
        // derives an effective IDD4R/IDD4W (mA) from the per-bus activity knobs. When
        // the model is disabled, these are the unchanged config IDD4R/IDD4W.
        double iDD4R_mA = dram.memSpec.memPowerSpec.iDD4R;
        double iDD4W_mA = dram.memSpec.memPowerSpec.iDD4W;
        const auto &dp = dram.memSpec.dataPattern;
        if (dp.enabled) {
            const double idd3n1_mA = dram.memSpec.memPowerSpec.iDD3N1;
            iDD4R_mA = dp.effective_current_mA(iDD4R_mA, idd3n1_mA);
            if (dp.apply_to_writes)
                iDD4W_mA = dp.effective_current_mA(iDD4W_mA, idd3n1_mA);
        }
        auto IDD4R   = iDD4R_mA * 1e-3;
        auto IDD4W   = iDD4W_mA * 1e-3;

        // For HBM2 IBeta = IDD2N (no explicit precharge operating current)
        auto IBeta = IDD2N;

        // Derived currents
        // I_rho: bank-wise weighted active standby current
        auto I_rho = rho * (IDD3N1 - IDD2N) + IDD2N;

        // I_theta: activation current derived from IDD0 cycle measurement
        auto I_theta = (IDD0 * (t_RP + t_RAS) - IBeta * t_RP) / t_RAS;

        // I_1: single-bank active standby current distributed across all banks
        auto I_1 = (1.0 / B) * (IDD3N1 + (B - 1) * I_rho);

        assert(I_rho >= 0 && "I_rho must be non-negative");
        assert(I_theta >= 0 && "I_theta must be non-negative");
        assert(I_1 >= 0 && "I_1 must be non-negative");

        energy_t energy(dram.memSpec.banksPerPseudoChannel);

        // Per-additional-active-bank current increment: going from one active bank
        // (IDD3N1) to all B active (IDD3N16) raises active-standby current by
        // (IDD3N16 - IDD3N1) spread over the (B-1) extra banks.
        const double delta_bg_act = (B > 1) ? (IDD3N16 - IDD3N1) / (B - 1) : 0.0;

        // Single pseudo-channel: rank index 0
        const auto &bgFactors  = dram.memSpec.memPowerSpec.bankgroup_scaling_factors;
        const auto &bankFactors = dram.memSpec.memPowerSpec.bank_scaling_factors;
        const std::size_t banksPerGroup = dram.memSpec.numberOfBanks; // banks per bank-group
        for (std::size_t b = 0; b < B; ++b) {
            const auto &bank = stats.bank[b];

            // Systematic structural variation: scale this bank's read/write current by
            // bankgroup_factor[b / banksPerGroup] * bank_factor[b % banksPerGroup].
            // Empty factor vectors => uniform (factor 1.0), preserving prior behaviour.
            double varFactor = 1.0;
            if (!bgFactors.empty())
                varFactor *= bgFactors[(b / banksPerGroup) % bgFactors.size()];
            if (!bankFactors.empty())
                varFactor *= bankFactors[(b % banksPerGroup) % bankFactors.size()];
            const double IDD4R_b = IDD4R * varFactor;
            const double IDD4W_b = IDD4W * varFactor;

            // Time this bank spent active (precharged-bank time handled by E_bg_pre below).
            const double T_bank_act = bank.cycles.activeTime() * t_CK;
            // Burst window per access = (BL / DR) data-rate cycles, in ps.
            const double t_burst = (double(BL) / DR) * t_CK;

            // Activation energy: excess activation current (I_theta) above the
            // distributed single-bank active standby (I_1), over t_RAS, per ACT.
            energy.bank_energy[b].E_act +=
                VDD * (I_theta - I_1) * t_RAS * bank.counter.act;

            // Precharge energy: excess precharge current (IBeta) above precharged
            // standby (IDD2N), over t_RP, per explicit PRE.
            energy.bank_energy[b].E_pre +=
                VDD * (IBeta - IDD2N) * t_RP * bank.counter.pre;

            // Active background energy, per bank: each active bank draws delta_bg_act
            // over its active time. Combined with the shared base term below, the first
            // active bank totals IDD3N1 and all B active total IDD3N16.
            energy.bank_energy[b].E_bg_act +=
                VDD * delta_bg_act * T_bank_act;

            // Precharged background energy: precharged standby current (IDD2N) over the
            // rank's precharged time, shared evenly across the B banks (1/B).
            energy.bank_energy[b].E_bg_pre +=
                (1.0 / B) * VDD * IDD2N * (stats.rank_total[0].cycles.pre * t_CK);

            // Read energy: excess read current (IDD4R) above active standby (IDD3N1),
            // over the burst window, per read burst.
            energy.bank_energy[b].E_RD +=
                VDD * (IDD4R_b - IDD3N1) * t_burst * bank.counter.reads;

            // Write energy: excess write current (IDD4W) above active standby (IDD3N1),
            // over the burst window, per write burst.
            energy.bank_energy[b].E_WR +=
                VDD * (IDD4W_b - IDD3N1) * t_burst * bank.counter.writes;

            // Read-with-autoprecharge: read-burst component (precharge component below).
            energy.bank_energy[b].E_RDA +=
                VDD * (IDD4R_b - IDD3N1) * t_burst * bank.counter.readAuto;

            // Write-with-autoprecharge: write-burst component (precharge component below).
            energy.bank_energy[b].E_WRA +=
                VDD * (IDD4W_b - IDD3N1) * t_burst * bank.counter.writeAuto;

            // Precharge component of read-with-autoprecharge (one PRE per RDA).
            energy.bank_energy[b].E_pre_RDA +=
                VDD * (IBeta - IDD2N) * t_RP * bank.counter.readAuto;

            // Precharge component of write-with-autoprecharge (one PRE per WRA).
            energy.bank_energy[b].E_pre_WRA +=
                VDD * (IBeta - IDD2N) * t_RP * bank.counter.writeAuto;
        }

        // Shared base active-standby energy: (IDD3N1 - delta_bg_act) over the time the
        // rank has >=1 bank active. The remaining delta_bg_act for the first active bank
        // is supplied by that bank's per-bank term above, so one active bank totals
        // IDD3N1 (and the IDD2N floor is carried through active time, not dropped).
        energy.E_bg_act_shared +=
            VDD * (IDD3N1 - delta_bg_act) * (stats.rank_total[0].cycles.act * t_CK);

        return energy;
    }

} // namespace DRAMPower
