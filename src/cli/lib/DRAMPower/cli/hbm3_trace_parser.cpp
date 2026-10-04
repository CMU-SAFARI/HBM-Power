#include "hbm3_trace_parser.hpp"

#include <string>

#include <spdlog/spdlog.h>

#include <DRAMPower/command/CmdType.h>

#include "csv.hpp"
#include "util.hpp"

namespace DRAMPower::DRAMPowerCLI {

using namespace DRAMPower;

bool parse_hbm3_trace(
    std::string_view csv_file,
    std::vector<HBM3TraceEntry> &entries,
    std::size_t banks_per_group)
{
    csv::CSVFormat format;
    format.no_header();
    format.trim({ ' ', '\t' });

    csv::CSVReader reader{ csv_file, format };

    // CSV columns:
    // timestamp, command, channel_id, pseudochannel_id, bankgroup_id, bank_id, row_id[, column_id]
    // column_id is optional for commands that don't use it (ACT, PRE, etc.)
    constexpr std::size_t MIN_COLUMNS = 7;

    uint64_t rowcounter = 0;

    for (csv::CSVRow &row : reader) {
        // Skip header row if present
        if (rowcounter == 0) {
            csv::string_view first = row[0].get_sv();
            if (first == "timestamp") {
                ++rowcounter;
                continue;
            }
        }

        if (row.size() < MIN_COLUMNS) {
            spdlog::error("HBM3 trace line {}: expected at least {} columns, got {}",
                rowcounter + 1, MIN_COLUMNS, row.size());
            return false;
        }

        std::size_t idx = 0;

        timestamp_t timestamp = row[idx++].get<timestamp_t>();
        csv::string_view cmdStr = row[idx++].get_sv();
        std::size_t channel_id = row[idx++].get<std::size_t>();
        std::size_t pseudochannel_id = row[idx++].get<std::size_t>();
        std::size_t bankgroup_id = row[idx++].get<std::size_t>();
        std::size_t bank_id = row[idx++].get<std::size_t>();
        std::size_t row_id = row[idx++].get<std::size_t>();

        CmdType cmd = CmdTypeUtil::from_string(cmdStr);

        // column_id is optional — ACT, PRE, etc. don't use it and traces often omit it
        std::size_t column_id = 0;
        if (idx < row.size()) {
            csv::string_view col_sv = row[idx].get_sv();
            if (!col_sv.empty()) {
                column_id = row[idx].get<std::size_t>();
            }
            ++idx;
        }

        // Flat bank index within the pseudo-channel
        std::size_t flat_bank = bankgroup_id * banks_per_group + bank_id;

        TargetCoordinate coord{flat_bank, bankgroup_id, 0, row_id, column_id};

        // Handle optional data payload for read/write commands
        // Data column is only present if the trace includes hex payload after column_id
        if (CmdTypeUtil::needs_data(cmd) && idx < row.size()) {
            csv::string_view data = row[idx].get_sv();
            if (!data.empty()) {
                std::size_t size = 0;
                std::unique_ptr<uint8_t[]> arr;
                try {
                    arr = util::hexStringToUint8Array(data, size);
                } catch (std::exception &e) {
                    spdlog::error("HBM3 trace line {}: failed to parse data hex string", rowcounter + 1);
                    return false;
                }
                ++idx;

                HBM3TraceEntry entry;
                entry.command = Command{timestamp, cmd, coord, arr.get(), size * 8};
                entry.data = std::move(arr);
                entry.channel_id = channel_id;
                entry.pseudochannel_id = pseudochannel_id;
                entries.push_back(std::move(entry));
            } else {
                HBM3TraceEntry entry;
                entry.command = Command{timestamp, cmd, coord};
                entry.data = nullptr;
                entry.channel_id = channel_id;
                entry.pseudochannel_id = pseudochannel_id;
                entries.push_back(std::move(entry));
            }
        } else {
            HBM3TraceEntry entry;
            entry.command = Command{timestamp, cmd, coord};
            entry.data = nullptr;
            entry.channel_id = channel_id;
            entry.pseudochannel_id = pseudochannel_id;
            entries.push_back(std::move(entry));
        }

        ++rowcounter;
    }

    return true;
}

bool parse_hbm3_command_list(
    std::string_view csv_file,
    std::vector<std::pair<Command, std::unique_ptr<uint8_t[]>>> &commandList,
    std::size_t banks_per_group)
{
    std::vector<HBM3TraceEntry> entries;
    if (!parse_hbm3_trace(csv_file, entries, banks_per_group)) {
        return false;
    }

    commandList.reserve(entries.size());
    for (auto &entry : entries) {
        commandList.emplace_back(std::move(entry.command), std::move(entry.data));
    }

    return true;
}

} // namespace DRAMPower::DRAMPowerCLI
