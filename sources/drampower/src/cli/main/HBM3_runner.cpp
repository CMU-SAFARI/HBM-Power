#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <optional>

#include <DRAMPower/cli/hbm3_trace_parser.hpp>
#include <DRAMPower/command/CmdType.h>
#include <DRAMPower/standards/hbm3/HBM3.h>
#include <DRAMPower/memspec/MemSpecHBM3.h>
#include <DRAMPower/data/energy.h>

#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

using json = nlohmann::json;
using namespace DRAMPower;

// Build MemSpecHBM3 from separate JSON configs
static MemSpecHBM3 buildMemSpec(const json &orgJson, const json &timingJson, const json &powerJson)
{
    // Architecture (HBM3 defaults: 16 channels/stack, 2 PC/ch, BL16, 32-bit PC)
    MemSpecHBM3::MemArchitectureSpec arch{};
    arch.nbrOfChannels        = orgJson.value("channels_per_stack", 16) * orgJson.value("stacks", 1);
    arch.nbrOfPseudoChannels  = orgJson.value("pseudochannels_per_channel", 2);
    arch.nbrOfBankGroups      = orgJson.value("bankgroups_per_pseudochannel", 4);
    arch.nbrOfBanks           = orgJson.value("banks_per_bankgroup", 4);
    arch.nbrOfRows            = orgJson.value("rows", 16384);
    arch.nbrOfColumns         = orgJson.value("columns", 64);
    arch.width                = orgJson.value("width", 32);
    arch.dataRate             = orgJson.value("dataRate", 2);

    const auto &t = timingJson["timing"];
    arch.burstLength = t.value("nBL", 16);

    // Timing
    MemSpecHBM3::MemTimingSpec timing{};
    timing.tCK     = t.value("tCK_ps", 0.0);
    timing.tRAS    = t.value("nRAS", 0);
    timing.tRCD    = t.value("nRCD", 0);
    timing.tRCDWR  = t.value("nRCDWR", 0);
    timing.tRC     = t.value("nRC", 0);
    timing.tRL     = t.value("nRL", 0);
    timing.tWL     = t.value("nWL", 0);
    timing.tCCD_S  = t.value("nCCD_S", 0);
    timing.tCCD_L  = t.value("nCCD_L", 0);
    timing.tWTR_S  = t.value("nWTR_S", 0);
    timing.tWTR_L  = t.value("nWTR_L", 0);
    timing.tWR     = t.value("nWR", 0);
    timing.tRP     = t.value("nRP", 0);
    // tBurst is computed in MemSpecHBM3 constructor from burstLength / dataRate

    // Power
    MemSpecHBM3::MemPowerSpec power{};
    const auto &v = powerJson["voltage"];
    const auto &idd = powerJson["IDD"];
    power.vDD     = v.value("VDD", 1.1);
    power.vDDQ    = v.value("VDDQ", power.vDD);   // I/O rail; defaults to VDD (single-rail)
    power.iDD0    = idd.value("IDD0", 0.0);
    power.iDD2N   = idd.value("IDD2N", 0.0);
    power.iDD3N1  = idd.value("IDD3N1", 0.0);
    power.iDD3N16 = idd.value("IDD3N16", 0.0);
    power.iDD4R   = idd.value("IDD4R", 0.0);
    power.iDD4W   = idd.value("IDD4W", 0.0);

    MemSpecHBM3 spec(arch, timing, power);

    // Optional data-pattern read/write energy model. HBM3 reuses the HBM2 sec.16
    // shape coefficients but drops cross-coupling (use_coupling stays false) and uses
    // the relative (K-free) form, scaling its own all-0s IDD4R/IDD4W by activity.
    if (powerJson.contains("datapattern")) {
        const auto &dp = powerJson["datapattern"];
        auto &m = spec.dataPattern;
        m.enabled        = dp.value("enabled", true);
        m.floor_pJbit    = dp.value("floor_pJbit", m.floor_pJbit);
        m.c_dq           = dp.value("coef_T_DQ", m.c_dq);
        m.c_t2bit        = dp.value("coef_T_2bit", m.c_t2bit);
        m.c_busflip      = dp.value("coef_busflip", m.c_busflip);
        m.use_coupling   = dp.value("use_coupling", m.use_coupling);
        m.c_bg_coupling  = dp.value("coef_BG_coupling", m.c_bg_coupling);
        m.c_tsv_coupling = dp.value("coef_TSV_coupling", m.c_tsv_coupling);
        m.bgcpl_full     = dp.value("bgcpl_full", m.bgcpl_full);
        m.tsvcpl_full    = dp.value("tsvcpl_full", m.tsvcpl_full);
        m.dq_rate        = dp.value("dq_rate", m.dq_rate);
        m.tsv_rate       = dp.value("tsv_rate", m.tsv_rate);
        m.bg_rate        = dp.value("bg_rate", m.bg_rate);
        m.ref_dq_rate    = dp.value("ref_dq_rate", m.ref_dq_rate);
        m.ref_tsv_rate   = dp.value("ref_tsv_rate", m.ref_tsv_rate);
        m.ref_bg_rate    = dp.value("ref_bg_rate", m.ref_bg_rate);
        m.K              = dp.value("K_mA_per_pJbit", m.K);
        m.apply_to_writes = dp.value("apply_to_writes", m.apply_to_writes);
    }

    return spec;
}

static json loadJson(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open: " + path);
    }
    return json::parse(ifs, nullptr, false, true);
}

// Extract optional data-pattern knob overrides (--dq-rate=, --tsv-rate=, --bg-rate=)
// from argv, leaving only positional arguments.
static void extractRateOverrides(int &argc, char *argv[],
        std::optional<double> &dq, std::optional<double> &tsv, std::optional<double> &bg)
{
    int w = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eat = [&](const char *key, std::optional<double> &dst) -> bool {
            std::string pfx = std::string(key) + "=";
            if (a.rfind(pfx, 0) == 0) { dst = std::stod(a.substr(pfx.size())); return true; }
            return false;
        };
        if (eat("--dq-rate", dq) || eat("--tsv-rate", tsv) || eat("--bg-rate", bg)) continue;
        argv[w++] = argv[i];
    }
    argc = w;
}

int main(int argc, char *argv[])
{
    std::optional<double> ovDqRate, ovTsvRate, ovBgRate;
    extractRateOverrides(argc, argv, ovDqRate, ovTsvRate, ovBgRate);

    if (argc < 5 || argc > 6) {
        spdlog::error("Usage: {} <organization.json> <timing.json> <power.json> <trace.csv> [variation.json] "
                      "[--dq-rate=R] [--tsv-rate=R] [--bg-rate=R]", argv[0]);
        return 1;
    }

    std::string orgFile   = argv[1];
    std::string timingFile = argv[2];
    std::string powerFile  = argv[3];
    std::string traceFile  = argv[4];
    std::string variationFile = (argc == 6) ? argv[5] : "";

    // Validate files exist
    std::vector<std::string> filesToCheck = {orgFile, timingFile, powerFile, traceFile};
    if (!variationFile.empty()) filesToCheck.push_back(variationFile);

    for (const auto &path : filesToCheck) {
        if (!std::filesystem::is_regular_file(path)) {
            spdlog::error("File not found: {}", path);
            return 1;
        }
    }

    spdlog::set_pattern("%v");

    // Load JSON configs
    json orgJson, timingJson, powerJson;
    try {
        orgJson    = loadJson(orgFile);
        timingJson = loadJson(timingFile);
        powerJson  = loadJson(powerFile);
    } catch (const std::exception &e) {
        spdlog::error("Error loading config: {}", e.what());
        return 1;
    }

    // Build memory specification
    MemSpecHBM3 memSpec = buildMemSpec(orgJson, timingJson, powerJson);

    // Apply CLI data-pattern knob overrides (enable the model if any is given).
    if (ovDqRate || ovTsvRate || ovBgRate) {
        memSpec.dataPattern.enabled = true;
        if (ovDqRate)  memSpec.dataPattern.dq_rate  = *ovDqRate;
        if (ovTsvRate) memSpec.dataPattern.tsv_rate = *ovTsvRate;
        if (ovBgRate)  memSpec.dataPattern.bg_rate  = *ovBgRate;
        spdlog::info("Data-pattern model: dq_rate={}, tsv_rate={}, bg_rate={} (use_coupling={}, K={})",
            memSpec.dataPattern.dq_rate, memSpec.dataPattern.tsv_rate, memSpec.dataPattern.bg_rate,
            memSpec.dataPattern.use_coupling, memSpec.dataPattern.K);
    }

    // Apply optional systematic (bank-group / bank) variation factors.
    // bankgroup_scaling_factors[bg] and bank_scaling_factors[bank_offset] scale the
    // per-bank read/write current; per-bank factor = bg_factor[b/banksPerGroup] * bank_factor[b%banksPerGroup].
    if (!variationFile.empty()) {
        try {
            json varJson = loadJson(variationFile);
            const auto &sf = varJson.at("scaling_factors");
            memSpec.memPowerSpec.bankgroup_scaling_factors =
                sf.at("bankgroup").get<std::vector<double>>();
            memSpec.memPowerSpec.bank_scaling_factors =
                sf.at("bank").get<std::vector<double>>();
            spdlog::info("Applied systematic variation from {}: {} bank-group, {} bank factors",
                variationFile,
                memSpec.memPowerSpec.bankgroup_scaling_factors.size(),
                memSpec.memPowerSpec.bank_scaling_factors.size());
        } catch (const std::exception &e) {
            spdlog::warn("Could not load/apply variation file: {}", e.what());
        }
    }

    spdlog::info("HBM3 Configuration:");
    spdlog::info("  Channels: {}, PseudoChannels/Ch: {}", memSpec.numberOfChannels, memSpec.numberOfPseudoChannels);
    spdlog::info("  BankGroups: {}, Banks/BG: {}, Banks/PCh: {}", memSpec.numberOfBankGroups, memSpec.numberOfBanks, memSpec.banksPerPseudoChannel);
    spdlog::info("  tCK={} ps, tRCD={}, tRAS={}, tRP={}, tRL={}, tWL={}", memSpec.memTimingSpec.tCK, memSpec.memTimingSpec.tRCD, memSpec.memTimingSpec.tRAS, memSpec.memTimingSpec.tRP, memSpec.memTimingSpec.tRL, memSpec.memTimingSpec.tWL);

    // Parse trace file
    std::size_t banks_per_group = memSpec.numberOfBanks;

    std::vector<DRAMPower::DRAMPowerCLI::HBM3TraceEntry> traceEntries;
    if (!DRAMPower::DRAMPowerCLI::parse_hbm3_trace(traceFile, traceEntries, banks_per_group)) {
        spdlog::error("Error while parsing trace file: {}", traceFile);
        return 1;
    }

    spdlog::info("Loaded {} commands from trace file", traceEntries.size());

    // Group trace entries by (channel, pseudochannel)
    // Key: (channel_id, pseudochannel_id)
    using PChKey = std::pair<std::size_t, std::size_t>;
    std::map<PChKey, std::vector<const DRAMPower::DRAMPowerCLI::HBM3TraceEntry*>> perPseudoChannel;

    for (const auto &entry : traceEntries) {
        perPseudoChannel[{entry.channel_id, entry.pseudochannel_id}].push_back(&entry);
    }

    spdlog::info("Trace spans {} active pseudo-channels", perPseudoChannel.size());

    // Simulate each pseudo-channel independently
    double totalEnergy = 0.0;
    timestamp_t maxElapsedCycles = 0;

    for (const auto &[key, entries] : perPseudoChannel) {
        auto [ch, pch] = key;

        // Create a fresh HBM3 instance per pseudo-channel
        HBM3 dram(memSpec);

        // Feed commands
        for (const auto *entry : entries) {
            dram.doCoreInterfaceCommand(entry->command);
        }

        // Calculate energy
        timestamp_t lastTime = dram.getLastCommandTime();
        energy_t coreEnergy = dram.calcCoreEnergy(lastTime);
        interface_energy_info_t ifEnergy = dram.calcInterfaceEnergy(lastTime);
        double pchTotal = coreEnergy.total() + ifEnergy.total();
        totalEnergy += pchTotal;

        if (lastTime > maxElapsedCycles)
            maxElapsedCycles = lastTime;

        // Per-pseudo-channel power
        double pchDuration_ps = static_cast<double>(lastTime) * memSpec.memTimingSpec.tCK;
        double pchPower_mW = (pchDuration_ps > 0.0) ? (pchTotal / pchDuration_ps) * 1e3 : 0.0;

        spdlog::info("Channel {} PCh {} : {} cmds, energy = {:.3f} pJ, power = {:.3f} mW",
            ch, pch, entries.size(), pchTotal, pchPower_mW);

        // Per-bank breakdown
        auto bankCount = dram.getBankCount();
        for (std::size_t b = 0; b < bankCount; ++b) {
            spdlog::info("  Bank {:2d} : {}", b, coreEnergy.bank_energy[b]);
        }
    }

    // Total average power
    double totalDuration_ps = static_cast<double>(maxElapsedCycles) * memSpec.memTimingSpec.tCK;
    double totalPower_mW = (totalDuration_ps > 0.0) ? (totalEnergy / totalDuration_ps) * 1e3 : 0.0;

    spdlog::info("Total energy across all pseudo-channels: {:.3f} pJ", totalEnergy);
    spdlog::info("Elapsed cycles: {}, duration: {:.3f} ns", maxElapsedCycles, totalDuration_ps / 1e3);
    spdlog::info("Average power: {:.3f} mW", totalPower_mW);

    return 0;
}
