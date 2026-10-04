#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <optional>

#include <DRAMPower/cli/hbm2_trace_parser.hpp>
#include <DRAMPower/command/CmdType.h>
#include <DRAMPower/standards/hbm2/HBM2.h>
#include <DRAMPower/memspec/MemSpecHBM2.h>
#include <DRAMPower/data/energy.h>

#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

using json = nlohmann::json;
using namespace DRAMPower;

// Build MemSpecHBM2 from separate JSON configs
static MemSpecHBM2 buildMemSpec(const json &orgJson, const json &timingJson, const json &powerJson)
{
    // Every field missing from the JSON files takes its value from config/HBM2_1200MTs
    // (the defaults below must match that directory).
    // Architecture
    MemSpecHBM2::MemArchitectureSpec arch{};
    arch.nbrOfChannels        = orgJson.value("channels_per_stack", 8) * orgJson.value("stacks", 1);
    arch.nbrOfPseudoChannels  = orgJson.value("pseudochannels_per_channel", 2);
    arch.nbrOfBankGroups      = orgJson.value("bankgroups_per_pseudochannel", 4);
    arch.nbrOfBanks           = orgJson.value("banks_per_bankgroup", 4);
    arch.nbrOfRows            = orgJson.value("rows", 16384);
    arch.nbrOfColumns         = orgJson.value("columns", 32);
    arch.width                = orgJson.value("width", 64);
    arch.dataRate             = orgJson.value("dataRate", 2);

    const json t = timingJson.value("timing", json::object());
    arch.burstLength = t.value("nBL", 4);

    // Timing
    MemSpecHBM2::MemTimingSpec timing{};
    timing.tCK     = t.value("tCK_ps", 1666.7);
    timing.tRAS    = t.value("nRAS", 20);
    timing.tRCD    = t.value("nRCD", 9);
    timing.tRCDWR  = t.value("nRCDWR", 6);
    timing.tRC     = t.value("nRC", 29);
    timing.tRL     = t.value("nRL", 7);
    timing.tWL     = t.value("nWL", 7);
    timing.tCCD_S  = t.value("nCCD_S", 2);
    timing.tCCD_L  = t.value("nCCD_L", 2);
    timing.tWTR_S  = t.value("nWTR_S", 3);
    timing.tWTR_L  = t.value("nWTR_L", 6);
    timing.tWR     = t.value("nWR", 10);
    timing.tRP     = t.value("nRP", 9);
    timing.tRFC    = t.value("nRFC", 210);        // all-bank refresh; 0 disables refresh energy
    // tBurst is computed in MemSpecHBM2 constructor from burstLength / dataRate

    // Power
    MemSpecHBM2::MemPowerSpec power{};
    const json v = powerJson.value("voltage", json::object());
    const json idd = powerJson.value("IDD", json::object());
    power.vDD     = v.value("VDD", 1.2);
    power.iDD0    = idd.value("IDD0", 28.544);
    power.iDD2N   = idd.value("IDD2N", 11.756);
    power.iDD3N1  = idd.value("IDD3N1", 12.394);
    power.iDD3N16 = idd.value("IDD3N16", 14.281);
    power.iDD4R   = idd.value("IDD4R", 202.769);
    power.iDD4W   = idd.value("IDD4W", 137.406);
    power.iDD5B   = idd.value("IDD5B", 66.331);   // 0 disables refresh energy

    MemSpecHBM2 spec(arch, timing, power);

    // Data-pattern model: enabled by default with the DataPatternModel defaults (the coefficients
    // and knobs of the device configurations); a "datapattern" block configures or disables it.
    spec.dataPattern.enabled = true;
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
    json j = json::parse(ifs, nullptr, false, true);
    if (j.is_discarded()) {
        throw std::runtime_error("Invalid JSON: " + path);
    }
    return j;
}

// Extract optional data-pattern knob overrides (--dq-rate=, --tsv-rate=, --bg-rate=)
// from argv, leaving only positional arguments. Lets users sweep activity without
// editing the power JSON.
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
    MemSpecHBM2 memSpec = buildMemSpec(orgJson, timingJson, powerJson);

    // Apply CLI data-pattern knob overrides (enable the model if any is given).
    if (ovDqRate || ovTsvRate || ovBgRate) {
        memSpec.dataPattern.enabled = true;
        if (ovDqRate)  memSpec.dataPattern.dq_rate  = *ovDqRate;
        if (ovTsvRate) memSpec.dataPattern.tsv_rate = *ovTsvRate;
        if (ovBgRate)  memSpec.dataPattern.bg_rate  = *ovBgRate;
        spdlog::info("Data-pattern model: dq_rate={}, tsv_rate={}, bg_rate={} ({} form)",
            memSpec.dataPattern.dq_rate, memSpec.dataPattern.tsv_rate, memSpec.dataPattern.bg_rate,
            memSpec.dataPattern.K > 0.0 ? "absolute" : "relative");
    }

    // Apply optional systematic (bank-group / bank) variation factors.
    // bankgroup_scaling_factors[bg] and bank_scaling_factors[bank_offset] scale the
    // per-bank read/write current; per-bank factor = bg_factor[b/banksPerGroup] * bank_factor[b%banksPerGroup].
    // Optional "bankgroup_write" / "bank_write" give separate factors for writes.
    if (!variationFile.empty()) {
        try {
            json varJson = loadJson(variationFile);
            const auto &sf = varJson.at("scaling_factors");
            memSpec.memPowerSpec.bankgroup_scaling_factors =
                sf.at("bankgroup").get<std::vector<double>>();
            memSpec.memPowerSpec.bank_scaling_factors =
                sf.at("bank").get<std::vector<double>>();
            if (sf.contains("bankgroup_write"))
                memSpec.memPowerSpec.bankgroup_write_scaling_factors =
                    sf.at("bankgroup_write").get<std::vector<double>>();
            if (sf.contains("bank_write"))
                memSpec.memPowerSpec.bank_write_scaling_factors =
                    sf.at("bank_write").get<std::vector<double>>();
            spdlog::info("Applied systematic variation from {}: {} bank-group, {} bank factors",
                variationFile,
                memSpec.memPowerSpec.bankgroup_scaling_factors.size(),
                memSpec.memPowerSpec.bank_scaling_factors.size());
        } catch (const std::exception &e) {
            spdlog::warn("Could not load/apply variation file: {}", e.what());
        }
    }

    spdlog::info("HBM2 Configuration:");
    spdlog::info("  Channels: {}, PseudoChannels/Ch: {}", memSpec.numberOfChannels, memSpec.numberOfPseudoChannels);
    spdlog::info("  BankGroups: {}, Banks/BG: {}, Banks/PCh: {}", memSpec.numberOfBankGroups, memSpec.numberOfBanks, memSpec.banksPerPseudoChannel);
    spdlog::info("  tCK={} ps, tRCD={}, tRAS={}, tRP={}, tRL={}, tWL={}", memSpec.memTimingSpec.tCK, memSpec.memTimingSpec.tRCD, memSpec.memTimingSpec.tRAS, memSpec.memTimingSpec.tRP, memSpec.memTimingSpec.tRL, memSpec.memTimingSpec.tWL);

    // Parse trace file
    std::size_t banks_per_group = memSpec.numberOfBanks;

    std::vector<DRAMPower::DRAMPowerCLI::HBM2TraceEntry> traceEntries;
    if (!DRAMPower::DRAMPowerCLI::parse_hbm2_trace(traceFile, traceEntries, banks_per_group)) {
        spdlog::error("Error while parsing trace file: {}", traceFile);
        return 1;
    }

    spdlog::info("Loaded {} commands from trace file", traceEntries.size());

    // Group trace entries by (channel, pseudochannel)
    // Key: (channel_id, pseudochannel_id)
    using PChKey = std::pair<std::size_t, std::size_t>;
    std::map<PChKey, std::vector<const DRAMPower::DRAMPowerCLI::HBM2TraceEntry*>> perPseudoChannel;

    for (const auto &entry : traceEntries) {
        perPseudoChannel[{entry.channel_id, entry.pseudochannel_id}].push_back(&entry);
    }

    spdlog::info("Trace spans {} active pseudo-channels", perPseudoChannel.size());

    // Simulate each pseudo-channel independently
    double totalEnergy = 0.0;
    timestamp_t maxElapsedCycles = 0;

    for (const auto &[key, entries] : perPseudoChannel) {
        auto [ch, pch] = key;

        // Create a fresh HBM2 instance per pseudo-channel
        HBM2 dram(memSpec);

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
