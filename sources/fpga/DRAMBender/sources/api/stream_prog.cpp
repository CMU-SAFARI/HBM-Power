#include "stream_prog.h"
#include "platform.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>

// ---------------------------------------------------------------
// Construction
// ---------------------------------------------------------------

StreamProgram::StreamProgram() {}

// ---------------------------------------------------------------
// Command-by-command interface
// ---------------------------------------------------------------

void StreamProgram::add_command(uint32_t encoded_cmd)
{
    commands_.push_back(encoded_cmd);
}

void StreamProgram::add_nop()
{
    commands_.push_back(SoftMCPlatform::encodeNOP());
}

void StreamProgram::add_nops(int n)
{
    for (int i = 0; i < n; i++)
        commands_.push_back(SoftMCPlatform::encodeNOP());
}

void StreamProgram::add_command(uint8_t cmd_type, uint16_t row, uint8_t col,
                                uint8_t bg, uint8_t bank, uint8_t pc, uint8_t ch)
{
    commands_.push_back(
        SoftMCPlatform::encodeCommand(cmd_type, row, col, bg, bank, pc, ch));
}

// ---------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------

static uint8_t csvCmdStringToType(const std::string& s)
{
    if (s == "ACT")  return SoftMCPlatform::CMD_ACT;
    if (s == "PRE")  return SoftMCPlatform::CMD_PRE;
    if (s == "PREA") return SoftMCPlatform::CMD_PREA;
    if (s == "REF")  return SoftMCPlatform::CMD_REF;
    if (s == "RD")   return SoftMCPlatform::CMD_RD;
    if (s == "RDA")  return SoftMCPlatform::CMD_RDA;
    if (s == "WR")   return SoftMCPlatform::CMD_WR;
    if (s == "WRA")  return SoftMCPlatform::CMD_WRA;
    return SoftMCPlatform::CMD_NOP;
}

bool StreamProgram::load_csv(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "StreamProgram::load_csv: cannot open " << csv_path << std::endl;
        return false;
    }

    // Skip header
    std::string line;
    std::getline(file, line);

    struct Entry {
        uint64_t timestamp;
        uint8_t  cmd_type;
        uint8_t  ch, pc, bg, bank;
        uint16_t row;
        uint8_t  col;
    };

    std::vector<Entry> entries;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        Entry e = {};

        std::getline(ss, token, ','); e.timestamp = std::stoull(token);
        std::getline(ss, token, ','); e.cmd_type  = csvCmdStringToType(token);
        std::getline(ss, token, ','); e.ch   = token.empty() ? 0 : (uint8_t)std::stoul(token);
        std::getline(ss, token, ','); e.pc   = token.empty() ? 0 : (uint8_t)std::stoul(token);
        std::getline(ss, token, ','); e.bg   = token.empty() ? 0 : (uint8_t)std::stoul(token);
        std::getline(ss, token, ','); e.bank = token.empty() ? 0 : (uint8_t)std::stoul(token);
        std::getline(ss, token, ','); e.row  = token.empty() ? 0 : (uint16_t)std::stoul(token);
        std::getline(ss, token, ','); e.col  = token.empty() ? 0 : (uint8_t)std::stoul(token);

        entries.push_back(e);
    }

    if (entries.empty()) return false;

    commands_.clear();
    uint64_t current_ts = entries[0].timestamp;
    size_t idx = 0;

    while (idx < entries.size()) {
        // Fill NOP slots for idle cycles
        while (current_ts < entries[idx].timestamp) {
            commands_.push_back(SoftMCPlatform::encodeNOP());
            current_ts++;
        }
        // Emit the command at this timestamp
        const auto& e = entries[idx];
        commands_.push_back(
            SoftMCPlatform::encodeCommand(e.cmd_type, e.row, e.col, e.bg, e.bank, e.pc, e.ch));
        idx++;
        current_ts++;
    }

    return true;
}

// ---------------------------------------------------------------
// Compressed CSV loading (for the BRAM replay engine)
// ---------------------------------------------------------------

bool StreamProgram::load_csv_entries(const std::string& csv_path,
                                     std::vector<TraceEntry>& out,
                                     uint64_t* max_timestamp_out)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "StreamProgram::load_csv_entries: cannot open " << csv_path << std::endl;
        return false;
    }

    out.clear();
    uint64_t max_ts = 0;

    // Skip header
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;

        uint64_t ts;
        uint8_t  cmd_type;
        uint8_t  ch, pc, bg, bank, col;
        uint16_t row;

        std::getline(ss, tok, ','); ts       = std::stoull(tok);
        std::getline(ss, tok, ','); cmd_type = csvCmdStringToType(tok);
        std::getline(ss, tok, ','); ch   = tok.empty() ? 0 : (uint8_t)std::stoul(tok);
        std::getline(ss, tok, ','); pc   = tok.empty() ? 0 : (uint8_t)std::stoul(tok);
        std::getline(ss, tok, ','); bg   = tok.empty() ? 0 : (uint8_t)std::stoul(tok);
        std::getline(ss, tok, ','); bank = tok.empty() ? 0 : (uint8_t)std::stoul(tok);
        std::getline(ss, tok, ','); row  = tok.empty() ? 0 : (uint16_t)std::stoul(tok);
        std::getline(ss, tok, ','); col  = tok.empty() ? 0 : (uint8_t)std::stoul(tok);

        // Skip NOP / unrecognised rows — the engine reconstructs idle cycles.
        if (cmd_type == SoftMCPlatform::CMD_NOP) continue;

        TraceEntry e;
        e.timestamp = ts;
        e.command   = SoftMCPlatform::encodeCommand(cmd_type, row, col, bg, bank, pc, ch);
        out.push_back(e);
        if (ts > max_ts) max_ts = ts;
    }

    if (max_timestamp_out) *max_timestamp_out = max_ts;
    return !out.empty();
}

// ---------------------------------------------------------------
// Clear
// ---------------------------------------------------------------

void StreamProgram::clear()
{
    commands_.clear();
    write_data_.clear();
    has_default_write_data_ = false;
}

// ---------------------------------------------------------------
// Write data interface
// ---------------------------------------------------------------

void StreamProgram::set_write_data(size_t cmd_index, const uint8_t* data_32bytes)
{
    std::array<uint8_t, 32> arr;
    std::memcpy(arr.data(), data_32bytes, 32);
    write_data_[cmd_index] = arr;
}

void StreamProgram::set_default_write_data(const uint8_t* data_32bytes)
{
    std::memcpy(default_write_data_.data(), data_32bytes, 32);
    has_default_write_data_ = true;
}

void StreamProgram::clear_default_write_data()
{
    has_default_write_data_ = false;
}

const uint8_t* StreamProgram::get_write_data(size_t cmd_index) const
{
    auto it = write_data_.find(cmd_index);
    if (it != write_data_.end())
        return it->second.data();
    return nullptr;
}

const uint8_t* StreamProgram::get_default_write_data() const
{
    return has_default_write_data_ ? default_write_data_.data() : nullptr;
}

bool StreamProgram::has_any_write_data() const
{
    return has_default_write_data_ || !write_data_.empty();
}

// ---------------------------------------------------------------
// Pretty print
// ---------------------------------------------------------------

const char* StreamProgram::cmd_type_str(uint8_t cmd_type)
{
    switch (cmd_type) {
    case SoftMCPlatform::CMD_NOP:  return "NOP";
    case SoftMCPlatform::CMD_ACT:  return "ACT";
    case SoftMCPlatform::CMD_PRE:  return "PRE";
    case SoftMCPlatform::CMD_PREA: return "PREA";
    case SoftMCPlatform::CMD_REF:  return "REF";
    case SoftMCPlatform::CMD_RD:   return "RD";
    case SoftMCPlatform::CMD_RDA:  return "RDA";
    case SoftMCPlatform::CMD_WR:   return "WR";
    case SoftMCPlatform::CMD_WRA:  return "WRA";
    default:                       return "???";
    }
}

std::string StreamProgram::decode_command(uint32_t w)
{
    uint8_t  cmd_type = (w >> 28) & 0xF;
    uint16_t row      = (w >> 14) & 0x3FFF;
    uint8_t  col      = (w >>  9) & 0x1F;
    uint8_t  bg       = (w >>  7) & 0x3;
    uint8_t  bank     = (w >>  5) & 0x3;
    uint8_t  pc       = (w >>  4) & 0x1;
    uint8_t  ch       = w & 0xF;

    const char* name = cmd_type_str(cmd_type);

    char buf[128];
    if (cmd_type == SoftMCPlatform::CMD_NOP) {
        snprintf(buf, sizeof(buf), "NOP");
    } else {
        snprintf(buf, sizeof(buf), "%-4s ch=%u pc=%u bg=%u bank=%u row=%-5u col=%u",
                 name, ch, pc, bg, bank, row, col);
    }
    return std::string(buf);
}

void StreamProgram::pretty_print() const
{
    std::cout << "StreamProgram: " << commands_.size() << " command slots" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Collapse consecutive NOP runs for readability
    size_t i = 0;
    while (i < commands_.size()) {
        uint32_t w = commands_[i];
        uint8_t cmd_type = (w >> 28) & 0xF;

        if (cmd_type == SoftMCPlatform::CMD_NOP) {
            // Count consecutive NOPs
            size_t nop_start = i;
            while (i < commands_.size() && ((commands_[i] >> 28) & 0xF) == SoftMCPlatform::CMD_NOP)
                i++;
            size_t count = i - nop_start;
            if (count == 1) {
                printf("%06zu: NOP\n", nop_start);
            } else {
                printf("%06zu: NOP x %zu\n", nop_start, count);
            }
        } else {
            printf("%06zu: %s\n", i, decode_command(w).c_str());
            i++;
        }
    }

    std::cout << std::string(60, '-') << std::endl;
}
