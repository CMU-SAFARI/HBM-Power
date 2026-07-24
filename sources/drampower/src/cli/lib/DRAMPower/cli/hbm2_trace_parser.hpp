#ifndef LIB_DRAMPOWERCLI_HBM2_TRACE_PARSER_H
#define LIB_DRAMPOWERCLI_HBM2_TRACE_PARSER_H

#include <string_view>
#include <vector>
#include <memory>

#include <DRAMPower/command/Command.h>

namespace DRAMPower::DRAMPowerCLI {

// Parsed HBM2 trace entry with full addressing
struct HBM2TraceEntry {
    Command command;
    std::unique_ptr<uint8_t[]> data;
    std::size_t channel_id;
    std::size_t pseudochannel_id;
};

// Parse an HBM2 CSV trace file.
// Expected CSV format (with or without header):
//   timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id
//
// The bank field in TargetCoordinate is set to the flat bank index:
//   bankgroup_id * banks_per_group + bank_id
//
// Parameters:
//   csv_file        - path to the CSV trace file
//   entries         - output vector of parsed trace entries
//   banks_per_group - number of banks per bank group (from MemSpecHBM2::numberOfBanks)
//
// Returns true on success, false on parse error.
bool parse_hbm2_trace(
    std::string_view csv_file,
    std::vector<HBM2TraceEntry> &entries,
    std::size_t banks_per_group
);

// Convenience overload that outputs in the same format as parse_command_list,
// discarding channel/pseudochannel information.
bool parse_hbm2_command_list(
    std::string_view csv_file,
    std::vector<std::pair<Command, std::unique_ptr<uint8_t[]>>> &commandList,
    std::size_t banks_per_group
);

} // namespace DRAMPower::DRAMPowerCLI

#endif /* LIB_DRAMPOWERCLI_HBM2_TRACE_PARSER_H */
