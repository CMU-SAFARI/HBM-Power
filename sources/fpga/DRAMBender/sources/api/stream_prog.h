#ifndef STREAM_PROG_H
#define STREAM_PROG_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class StreamProgram {
public:
    StreamProgram();

    // ---------------------------------------------------------------
    // Command-by-command interface
    // ---------------------------------------------------------------

    /** Push a single encoded 32-bit command. */
    void add_command(uint32_t encoded_cmd);

    /** Push a NOP (idle cycle). */
    void add_nop();

    /** Push N consecutive NOPs. */
    void add_nops(int n);

    /** Encode and push a command from field values. */
    void add_command(uint8_t cmd_type, uint16_t row, uint8_t col,
                     uint8_t bg, uint8_t bank, uint8_t pc, uint8_t ch);

    // ---------------------------------------------------------------
    // Write data interface
    // ---------------------------------------------------------------

    /** Set 256 bits (32 bytes) of write data for the command at index
     *  @p cmd_index.  The command should be WR or WRA.
     *  The 256-bit word will be replicated to both pseudo-channel
     *  halves of the 512-bit HBM data bus in hardware. */
    void set_write_data(size_t cmd_index, const uint8_t* data_32bytes);

    /** Set a default write-data pattern applied to every WR/WRA command
     *  that does not have explicit per-command data.  If no default is
     *  set, commands without per-command data use all-zeros. */
    void set_default_write_data(const uint8_t* data_32bytes);

    /** Clear the default write-data pattern. */
    void clear_default_write_data();

    /** Get per-command write data, or nullptr if not set. */
    const uint8_t* get_write_data(size_t cmd_index) const;

    /** Get the default write-data pattern, or nullptr if not set. */
    const uint8_t* get_default_write_data() const;

    /** True if any per-command or default write data has been set. */
    bool has_any_write_data() const;

    // ---------------------------------------------------------------
    // CSV loading interface
    // ---------------------------------------------------------------

    /** Load commands from a CSV trace file (replaces current contents).
     *  CSV header: timestamp,command,channel_id,pseudochannel_id,
     *              bankgroup_id,bank_id,row_id,column_id
     *  Returns true on success. */
    bool load_csv(const std::string& csv_path);

    // ---------------------------------------------------------------
    // Compressed trace interface (for the BRAM replay engine)
    // ---------------------------------------------------------------

    /** One non-NOP command with its absolute timestamp (slot index). */
    struct TraceEntry {
        uint64_t timestamp;   // absolute slot index from the CSV 'timestamp' column
        uint32_t command;     // 32-bit encoded command (same layout as add_command)
    };

    /** Load a CSV trace in COMPRESSED form: one entry per non-NOP command, each
     *  with its absolute timestamp and 32-bit encoding. NOPs are NOT materialised
     *  (the replay engine reconstructs idle cycles from the cycle counter).
     *  @p out             receives the entries in CSV order (ascending timestamp).
     *  @p max_timestamp_out (optional) receives the largest timestamp seen.
     *  Returns true on success. */
    static bool load_csv_entries(const std::string& csv_path,
                                 std::vector<TraceEntry>& out,
                                 uint64_t* max_timestamp_out = nullptr);

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    /** Return the encoded command vector (read-only). */
    const std::vector<uint32_t>& commands() const { return commands_; }

    /** Number of command slots. */
    size_t size() const { return commands_.size(); }

    /** Clear all commands and write data. */
    void clear();

    // ---------------------------------------------------------------
    // Pretty print
    // ---------------------------------------------------------------

    /** Print the full trace to stdout in human-readable form. */
    void pretty_print() const;

private:
    std::vector<uint32_t> commands_;
    std::unordered_map<size_t, std::array<uint8_t, 32>> write_data_;
    std::array<uint8_t, 32> default_write_data_;
    bool has_default_write_data_ = false;

    /** Decode a 32-bit encoded command to a human-readable string. */
    static std::string decode_command(uint32_t w);

    /** Map a CMD_* constant to a short string. */
    static const char* cmd_type_str(uint8_t cmd_type);
};

#endif // STREAM_PROG_H
