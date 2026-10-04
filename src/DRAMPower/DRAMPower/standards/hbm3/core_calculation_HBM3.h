#ifndef DRAMPOWER_STANDARDS_HBM3_CALCULATION_HBM3_H
#define DRAMPOWER_STANDARDS_HBM3_CALCULATION_HBM3_H

#include <DRAMPower/data/energy.h>
#include <DRAMPower/Types.h>

#include <cstddef>
#include <cstdint>

namespace DRAMPower {

class HBM3;

class Calculation_HBM3
{
private:
        double E_act(double VDD, double I_theta, double I_1, double t_RAS, uint64_t N_act);
        double E_pre(double VDD, double IBeta, double IDD2_N, double t_RP, uint64_t N_pre);
        double E_BG_pre(std::size_t B, double VDD, double IDD2_N, double T_BG_pre);

    // Read energy: excess current during burst read
    double E_RD(double VDD, double IDD4R, double IDD3N, double t_CK, std::size_t BL, std::size_t DR, uint64_t N_RD);

    // Write energy: excess current during burst write
    double E_WR(double VDD, double IDD4W, double IDD3N, double t_CK, std::size_t BL, std::size_t DR, uint64_t N_WR);
    double E_ref_ab(std::size_t B, double VDD, double IDD5B, double IDD3N16, double tRFC, uint64_t N_REF);

public:
    energy_t calcEnergy(timestamp_t timestamp, HBM3 &dram);
};

} // namespace DRAMPower

#endif /* DRAMPOWER_STANDARDS_HBM3_CALCULATION_HBM3_H */
