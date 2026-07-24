#ifndef DRAMPOWER_STANDARDS_HBM2_CALCULATION_HBM2_H
#define DRAMPOWER_STANDARDS_HBM2_CALCULATION_HBM2_H

#include <DRAMPower/data/energy.h>
#include <DRAMPower/Types.h>

#include <cstddef>
#include <cstdint>

namespace DRAMPower {

class HBM2;

class Calculation_HBM2
{
public:
    // Computes the per-component energy for the window ending at `timestamp`.
    // All per-bank/background energy equations are inlined in the definition so
    // each component's formula is visible in one place.
    energy_t calcEnergy(timestamp_t timestamp, HBM2 &dram);
};

} // namespace DRAMPower

#endif /* DRAMPOWER_STANDARDS_HBM2_CALCULATION_HBM2_H */
