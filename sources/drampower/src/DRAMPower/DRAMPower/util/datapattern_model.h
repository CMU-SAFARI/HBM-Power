#ifndef DRAMPOWER_UTIL_DATAPATTERN_MODEL_H
#define DRAMPOWER_UTIL_DATAPATTERN_MODEL_H

// Data-pattern read/write energy model (HBM2/HBM3).
//
// Replaces the fixed IDD4R/IDD4W used by the core read/write energy with an
// effective current derived from the data activity on three physical buses, each
// driven by its own aggregate "toggle-rate" knob in [0,1] (0 = quiescent / static
// data, 1 = maximally toggling for that bus):
//
//   DQ   knob -> DQ-pin (I/O) toggle              : T_DQ
//   TSV  knob -> 2-bit on-die toggle (+ TSV xtalk) : T_2bit (+ TSV coupling)
//   BG   knob -> burst-to-burst inversion (+ BG xtalk): busflip (+ BG coupling)
//
// The per-read data-movement energy (pJ/bit) is the HBM2-fit linear model of
// record (notebook reproduce_comparison.ipynb, sec.16):
//
//   pJ/bit = floor + dq*Delta_DQ + tsv*Delta_TSV + bg*Delta_BG
//   Delta_DQ  = c_dq
//   Delta_TSV = c_t2bit  (+ c_tsv_coupling * tsvcpl_full   if use_coupling)
//   Delta_BG  = c_busflip(+ c_bg_coupling  * bgcpl_full    if use_coupling)
//
// Two ways to turn pJ/bit into an effective current (mA):
//   * absolute   (K > 0):  IDD4R_eff = IDD3N1 + K * pJ/bit
//                          Reproduces the HBM2 calibration exactly (K = 512 mA per
//                          pJ/bit there). K is tied to the bus width / measurement
//                          scale, so it does NOT transfer between standards.
//   * relative   (K == 0): IDD4R_eff = IDD3N1 + (IDD4R_base - IDD3N1) * S
//                          S = pJ/bit(knobs) / pJ/bit(reference). Dimensionless, so
//                          the SAME shape coefficients work for HBM2 and HBM3 (each
//                          scales its own dynamic read current). The config IDD4R is
//                          taken as the current at the reference activity (default
//                          quiescent, all knobs = 0 -> S = 1 -> unchanged).
// The relative form reproduces the absolute form exactly when the baseline IDD4R is
// the quiescent read current (IDD3N1 + K*floor), so it is a strict generalization.
//
// Defaults are the HBM2 sec.16 coefficients; HBM3 reuses them with use_coupling=false.

#include <cstddef>

namespace DRAMPower::util {

struct DataPatternModel {
    bool   enabled        = false;

    // pJ/bit shape (HBM2 model of record; reused across standards)
    double floor_pJbit    = 2.759;   // c0
    double c_dq           = 1.192;   // c1  (T_DQ)
    double c_t2bit        = 1.331;   // c2  (2-bit on-die toggle)
    double c_busflip      = 0.720;   // c3  (burst-to-burst inversion)
    bool   use_coupling   = false;   // HBM2: true ; HBM3: false
    double c_bg_coupling  = 0.345;   // c4
    double c_tsv_coupling = 0.078;   // c5
    double bgcpl_full     = 4.0;     // BG coupling at full bus activity (max squared diff)
    double tsvcpl_full    = 4.0;     // TSV coupling at full bus activity

    // Aggregate activity knobs (0..1), one per physical bus.
    double dq_rate        = 0.0;
    double tsv_rate       = 0.0;
    double bg_rate        = 0.0;

    // Reference activity that the baseline IDD4R/IDD4W in the config corresponds to
    // (relative form only). Default = quiescent.
    double ref_dq_rate    = 0.0;
    double ref_tsv_rate   = 0.0;
    double ref_bg_rate    = 0.0;

    // Optional absolute scale (mA per pJ/bit). >0 selects the absolute form.
    double K              = 0.0;

    bool   apply_to_writes = true;

    double delta_dq()  const { return c_dq; }
    double delta_tsv() const { return c_t2bit  + (use_coupling ? c_tsv_coupling * tsvcpl_full : 0.0); }
    double delta_bg()  const { return c_busflip + (use_coupling ? c_bg_coupling  * bgcpl_full  : 0.0); }

    double pj(double dq, double tsv, double bg) const {
        return floor_pJbit + dq * delta_dq() + tsv * delta_tsv() + bg * delta_bg();
    }
    double pj_knobs() const { return pj(dq_rate, tsv_rate, bg_rate); }
    double pj_ref()   const { return pj(ref_dq_rate, ref_tsv_rate, ref_bg_rate); }

    // Fraction of the dynamic read/write energy attributable to the external DQ pins,
    // i.e. the only component physically powered by the I/O rail (VDDQ). The on-die
    // floor, TSV (2-bit on-die toggle / TSV xtalk) and BG (burst inversion / BG xtalk)
    // components stay on the core rail (VDDC). Both the absolute and relative forms make
    // the dynamic current proportional to pj_knobs(), so the DQ share is the DQ term's
    // share of pj_knobs(). Returns 0 when there is no modelled activity.
    double dq_energy_fraction() const {
        const double tot = pj_knobs();
        return (tot > 0.0) ? (dq_rate * delta_dq()) / tot : 0.0;
    }

    // Effective current [mA] for a given baseline current and read background, applying
    // either the absolute (K) or relative (scaling) form. Used for both reads and writes.
    double effective_current_mA(double base_mA, double idd3n1_mA) const {
        if (K > 0.0) {
            return idd3n1_mA + K * pj_knobs();
        }
        const double ref = pj_ref();
        const double s = (ref != 0.0) ? (pj_knobs() / ref) : 1.0;
        return idd3n1_mA + (base_mA - idd3n1_mA) * s;
    }
};

} // namespace DRAMPower::util

#endif /* DRAMPOWER_UTIL_DATAPATTERN_MODEL_H */
