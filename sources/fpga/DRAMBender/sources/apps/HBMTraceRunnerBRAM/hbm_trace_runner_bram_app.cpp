// ===========================================================================
// HBMTraceRunnerBRAM
// ===========================================================================
//
// Loads a compressed HBM command trace into the FPGA's on-chip BRAM/URAM and
// replays it in a hardware loop with ZERO PCIe traffic during the run (companion
// to HBMTraceRunner, which streams the NOP-materialised command list from the
// host every iteration).
//
//   1. Reads a CSV trace (same format as cmd_traces/*.csv).
//   2. Compresses it to (timestamp, command) entries (drops NOPs).
//   3. Loads it into BRAM via loadTraceAndRun() and starts the loop.
//   4. The engine cycles through the trace until either
//        a) the requested --iterations count is reached, or
//        b) Ctrl+C requests a graceful stop (finishes the current iteration).
//   5. Reads back the FPGA cycle count and reports throughput.
//
// The on-chip engine uses one 64-bit slot counter advancing 4 slots/fab cycle;
// the trace is split into 4 banks by (timestamp mod 4) so it reproduces the
// exact command timing of stream mode (600 M slots/s).  See bram_replay.v.
// ===========================================================================

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <getopt.h>
#include <sys/stat.h>
#include <csignal>
#include <unistd.h>

#include "platform.h"
#include "stream_prog.h"

using namespace std;

static const double FAB_CLK_MHZ = 150.0;
// Must match bram_replay.v BANK_DEPTH (entries per bank). NOTE: this is the
// capacity of a bitstream built from the current bram_replay.v. A deployed
// bitstream built with the old value (65536) only accepts traces whose heaviest
// bank fits 65536; load the *_trunc traces there until the larger bitstream is
// flashed.
static const size_t BANK_DEPTH = 131072;

static const char* DEFAULT_TEST_PATH = "../../../../cmd_traces/llm_hbm2_40gb/stack0/pc0.csv";

static SoftMCPlatform*    g_platform = nullptr;
static std::atomic<bool>  g_monitoring_running{false};
static std::thread        g_monitor_thread;
static std::atomic<bool>  g_stop_requested{false};
static std::atomic<int>   g_sigint_count{0};

static void stop_monitoring()
{
    if (g_monitoring_running.exchange(false)) {
        if (g_monitor_thread.joinable())
            g_monitor_thread.join();
    }
}

// First Ctrl+C: request a graceful stop (the engine finishes the current
// iteration). Second Ctrl+C: hard abort.
static void sigint_handler(int)
{
    int n = ++g_sigint_count;
    if (n == 1) {
        const char* msg = "\n[SIGINT] graceful stop requested (finishing current iteration; Ctrl+C again to abort)\n";
        ssize_t r = write(STDERR_FILENO, msg, strlen(msg)); (void)r;
        g_stop_requested.store(true);   // watcher thread issues the stop (signal-safe)
    } else {
        const char* msg = "\n[SIGINT] aborting\n";
        ssize_t r = write(STDERR_FILENO, msg, strlen(msg)); (void)r;
        _exit(1);
    }
}

static string derive_trace_name(const string& csv_path)
{
    string name = csv_path;
    size_t slash = name.find_last_of('/');
    if (slash != string::npos) name = name.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != string::npos) name = name.substr(0, dot);
    return name;
}

// Parse "all" (0-15), "low" (0-7), "high" (8-15), or a comma list e.g. "0,1,2,3"
static vector<int> parse_channels(const string& mode)
{
    vector<int> ch;
    if (mode == "all")  { for (int i = 0; i < 16; i++) ch.push_back(i); }
    else if (mode == "low")  { for (int i = 0; i < 8;  i++) ch.push_back(i); }
    else if (mode == "high") { for (int i = 8; i < 16; i++) ch.push_back(i); }
    else {
        size_t start = 0;
        while (start < mode.size()) {
            size_t comma = mode.find(',', start);
            string tok = mode.substr(start, comma == string::npos ? string::npos : comma - start);
            if (!tok.empty()) ch.push_back(atoi(tok.c_str()));
            if (comma == string::npos) break;
            start = comma + 1;
        }
    }
    return ch;
}

static void printUsage(const char* prog)
{
    cerr << "Usage: " << prog << " [options]\n"
         << "  -h, --help            Show this help\n"
         << "  --csv PATH            CSV trace to load (default: " << DEFAULT_TEST_PATH << ")\n"
         << "  --iterations N        Loop iterations; 0 = run until Ctrl+C (default: 0)\n"
         << "  --channels MODE       'all' (0-15, default), 'low', 'high', or list '0,1,2'\n"
         << "  --period-ms N         HBM metrics sampling interval in ms (default: 1000)\n"
         << "  --output PATH         Metrics CSV path (default: results/<trace>_data_<wrpattern>_bram.csv)\n"
         << "  --no-garbage-reads    Do NOT drop HBM read data (default: garbage reads ON;\n"
         << "                        required so RD-heavy loops don't back up the C2H path)\n"
         << "  --wr-pattern HEX      32-bit write-data pattern for WR/WRA (default: DEADBEEF)\n"
         << "  --init MODE           Pre-write the trace's footprint before the loop (via the\n"
         << "                        stream path): 'none' (default), 'zeros', or 'random'\n"
         << "                        ('random' = a different 256-bit word per address, so\n"
         << "                        replayed reads return per-word random data)\n"
         << "  --init-seed N         Seed for --init random (default: 1; makes the data reproducible)\n"
         << "  --tail-gap N          Idle slots appended after the last command so the loop\n"
         << "                        seam leaves >= tRP before re-opening rows (default: 64)\n"
         << "  --verify              Run a write->read-back self-test (no CSV) and exit:\n"
         << "                        writes a pattern to one address, reads it back, compares\n"
         << "  --verify-stream       Same self-test but via the PCIe stream path (isolates\n"
         << "                        the shared readback path; combine with --verify to compare)\n";
}

// Dump up to 6 returned entries and report the best 32-byte entry's match
// against `pat`. PASS if some entry matches in >= min_words of its 8 words
// (tolerating an occasional marginal DRAM bit). The cycle-count block
// ({count,0,...}) and zeros score 0, so they never produce a false pass.
static bool check_and_dump(const std::vector<uint8_t>& rd, uint32_t pat, int min_words = 7)
{
    size_t entries = rd.size() / 32;
    int best = 0;
    for (size_t e = 0; e < entries; e++) {
        int m = 0;
        for (int w = 0; w < 8; w++) {
            uint32_t x; memcpy(&x, &rd[e*32 + w*4], 4);
            if (x == pat) m++;
        }
        if (m > best) best = m;
    }
    size_t dump = entries < 6 ? entries : 6;
    for (size_t e = 0; e < dump; e++) {
        cout << "      [" << e << "] ";
        for (int w = 0; w < 8; w++) {
            uint32_t x; memcpy(&x, &rd[e*32 + w*4], 4);
            cout << setw(8) << setfill('0') << hex << x << " ";
        }
        cout << dec << setfill(' ') << endl;
    }
    bool ok = best >= min_words;
    cout << "      -> best entry matches " << best << "/8 words of 0x"
         << hex << pat << dec << (ok ? "  (OK)" : "  (FAIL)") << endl;
    return ok;
}

// Write -> read-back self-test for the BRAM replay engine. Builds a tiny
// ACT/WR/PRE/ACT/RD/PRE trace on a single channel/bank, runs ONE iteration with
// garbage reads OFF (so the RD data returns over C2H), and checks the read data
// equals the written pattern. Repeats with two patterns so a match can't be a
// coincidence. Returns 0 on PASS, 1 on FAIL.
static int run_verify(SoftMCPlatform& platform, int vch)
{
    cout << "\n=== BRAM-replay self-test: write -> read-back (channel " << vch << ") ===" << endl;
    platform.set_garbage_reads(false);            // we want the HBM read data back
    platform.set_broadcast_channels({vch});       // single channel -> unambiguous read

    // One HBM address (bank fields all 0). The 4-way replay split is on the
    // timestamp and is unrelated to these HBM bank/row/col fields.
    const uint8_t  PC = 0, BG = 0, BANK = 0, CH = (uint8_t)vch, COL = 0;
    const uint16_t ROW = 0;

    // ACT, WR, PRE, ACT, RD, PRE — generous gaps (cycles) for tRCD/tWR/tRP/tRAS.
    const uint64_t ts_arr[6]   = { 0, 40, 90, 140, 180, 230 };
    const uint8_t  type_arr[6] = { SoftMCPlatform::CMD_ACT, SoftMCPlatform::CMD_WR,
                                   SoftMCPlatform::CMD_PRE, SoftMCPlatform::CMD_ACT,
                                   SoftMCPlatform::CMD_RD,  SoftMCPlatform::CMD_PRE };
    uint64_t timestamps[6];
    uint32_t commands[6];
    for (int i = 0; i < 6; i++) {
        timestamps[i] = ts_arr[i];
        commands[i]   = SoftMCPlatform::encodeCommand(type_arr[i], ROW, COL, BG, BANK, PC, CH);
    }
    // Big tail after the RD so the HBM read data (which takes ~30-45 fab
    // cycles to round-trip through the CDCs + PHY) lands in the readback FIFO
    // before the end-of-iteration flush ships the packet. The readback sender
    // also flushes every ~300 cycles, so a >300-fab-cycle (>1200-slot) tail
    // guarantees the read data is sent. (The normal looping path drops read
    // data via garbage-reads, so it doesn't need this.)
    uint64_t trace_len = ts_arr[5] + 1 + 2048;
    if (trace_len % 4) trace_len += 4 - (trace_len % 4);

    const uint32_t patterns[2] = { 0xA5A5A5A5u, 0x5A5A5A5Au };
    bool all_ok = true;
    for (int p = 0; p < 2; p++) {
        uint32_t pat = patterns[p];
        if (platform.loadTraceAndRun(timestamps, commands, 6, trace_len,
                                     /*iterations*/1, pat, /*wr_data_enable*/true) != 0) {
            cerr << "  loadTraceAndRun failed" << endl;
            return 1;
        }
        // Raw capture (do NOT auto-strip the cycle-count block — it would
        // mis-fire on the uniform write pattern). We instead scan for a full
        // 32-byte (8-word) entry that is entirely `pat`; the {8 x cycle_count}
        // flush block can never look like that, so there are no false matches.
        vector<uint8_t> rd;
        platform.waitTraceDoneCapture(rd, /*strip_cycle_count*/false);

        cout << "  pattern 0x" << hex << pat << dec << ": "
             << rd.size() << " bytes (" << rd.size()/32 << " entries)"
             << (rd.empty() ? "  NO DATA" : "") << endl;
        all_ok = all_ok && check_and_dump(rd, pat);
    }
    cout << "SELF-TEST: " << (all_ok ? "PASS" : "FAIL") << endl;
    if (!all_ok)
        cout << "  (if NO DATA: check the bitstream includes bram_replay, that this\n"
                "   channel is populated, and that garbage-reads is off in the run)" << endl;
    return all_ok ? 0 : 1;
}

// Same write->read-back test but via the EXISTING PCIe stream path
// (prepareStreamCommands + streamExecute, which captures read data). Lets us
// isolate whether the shared readback path returns HBM read data at all on this
// bitstream, vs a bram_replay-specific problem.
static int run_verify_stream(SoftMCPlatform& platform, int vch)
{
    cout << "\n=== STREAM-path self-test: write -> read-back (channel " << vch << ") ===" << endl;
    platform.set_garbage_reads(false);
    platform.set_broadcast_channels({vch});

    const uint8_t  PC = 0, BG = 0, BANK = 0, CH = (uint8_t)vch, COL = 0;
    const uint16_t ROW = 0;
    const uint32_t pat = 0xA5A5A5A5u;

    StreamProgram prog;
    uint8_t wd[32];
    for (int i = 0; i < 8; i++) memcpy(wd + i*4, &pat, 4);
    prog.set_default_write_data(wd);                 // WR uses this 32-byte pattern
    prog.add_command(SoftMCPlatform::CMD_ACT, ROW, COL, BG, BANK, PC, CH); prog.add_nops(20);
    prog.add_command(SoftMCPlatform::CMD_WR,  ROW, COL, BG, BANK, PC, CH); prog.add_nops(40);
    prog.add_command(SoftMCPlatform::CMD_PRE, ROW, COL, BG, BANK, PC, CH); prog.add_nops(20);
    prog.add_command(SoftMCPlatform::CMD_ACT, ROW, COL, BG, BANK, PC, CH); prog.add_nops(20);
    prog.add_command(SoftMCPlatform::CMD_RD,  ROW, COL, BG, BANK, PC, CH); prog.add_nops(60);
    prog.add_command(SoftMCPlatform::CMD_PRE, ROW, COL, BG, BANK, PC, CH); prog.add_nops(100);
    while (prog.size() % 8) prog.add_nop();

    size_t total = 0;
    void* buf = platform.prepareStreamCommands(prog, &total);
    if (!buf) { cerr << "  prepareStreamCommands failed" << endl; return 1; }
    vector<uint8_t> rd; uint64_t cyc = 0;
    int rc = platform.streamExecute(buf, total, rd, cyc);
    free(buf);
    if (rc) { cerr << "  streamExecute failed" << endl; return 1; }

    cout << "  stream read-back: " << rd.size() << " bytes (" << rd.size()/32
         << " entries), cyc=" << cyc << endl;
    bool found = check_and_dump(rd, pat);
    cout << "  STREAM SELF-TEST: " << (found ? "PASS" : "FAIL") << endl;
    return found ? 0 : 1;
}

// SplitMix64 — reproducible per-address PRNG for the random data-init pass.
static inline uint64_t splitmix64(uint64_t& x)
{
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Data initialisation pass.
//
// Before the measured BRAM loop, pre-write the trace's read/write footprint
// over the PCIe stream path (which supports per-command write data, unlike the
// BRAM engine's single broadcast pattern), so the replayed RD commands return
// the data we put there. With `randomize`, each address gets a DIFFERENT random
// 256-bit word; otherwise every address is written all-zeros. Both modes touch
// the exact same addresses, so a zeros-vs-random comparison differs only in the
// stored data value.
//
// Addresses are taken straight from the loaded trace (every RD/WR carries its
// full row+col), grouped by row so each row is opened once (ACT), all its
// columns written, then closed (PRE). `seed` makes the random data reproducible;
// the per-address value depends only on (seed, address), not on ordering.
//
// The write broadcasts to whatever channels are currently enabled (set by the
// caller via set_broadcast_channels), matching the channels the loop reads.
// ---------------------------------------------------------------------------
static int run_init_data(SoftMCPlatform& platform,
                         const std::vector<StreamProgram::TraceEntry>& entries,
                         bool randomize, uint32_t seed)
{
    // packed(pc,bg,bank,row) -> set of column addresses touched
    std::map<uint32_t, std::set<uint8_t>> row_cols;
    for (const auto& e : entries) {
        uint8_t type = (e.command >> 28) & 0xF;
        if (type != SoftMCPlatform::CMD_RD  && type != SoftMCPlatform::CMD_RDA &&
            type != SoftMCPlatform::CMD_WR  && type != SoftMCPlatform::CMD_WRA)
            continue;
        uint16_t row  = (e.command >> 14) & 0x3FFF;
        uint8_t  col  = (e.command >> 9)  & 0x1F;
        uint8_t  bg   = (e.command >> 7)  & 0x3;
        uint8_t  bank = (e.command >> 5)  & 0x3;
        uint8_t  pc   = (e.command >> 4)  & 0x1;
        uint32_t rkey = ((uint32_t)pc << 18) | ((uint32_t)bg << 16)
                      | ((uint32_t)bank << 14) | row;
        row_cols[rkey].insert(col);
    }

    size_t n_rows = row_cols.size(), n_addr = 0;
    for (auto& kv : row_cols) n_addr += kv.second.size();
    cout << "Init-data (" << (randomize ? "random" : "zeros") << "): writing "
         << n_addr << " addresses across " << n_rows << " rows";
    if (randomize) cout << " (seed=0x" << hex << seed << dec << ")";
    cout << "..." << endl;

    // Build + flush the write program in chunks to bound host memory.
    const size_t FLUSH_CMDS = 100000;
    StreamProgram prog;
    size_t total_written = 0;

    auto flush = [&]() -> int {
        if (prog.size() == 0) return 0;
        while (prog.size() % 8) prog.add_nop();        // beat alignment
        size_t total = 0;
        void* buf = platform.prepareStreamCommands(prog, &total);
        if (!buf) { cerr << "  init: prepareStreamCommands failed" << endl; return 1; }
        vector<uint8_t> rd; uint64_t cyc = 0;
        int rc = platform.streamExecute(buf, total, rd, cyc);
        free(buf);
        prog = StreamProgram();                        // reset for next chunk
        if (rc) { cerr << "  init: streamExecute failed (rc=" << rc << ")" << endl; return 1; }
        return 0;
    };

    for (auto& kv : row_cols) {
        uint32_t rkey = kv.first;
        uint16_t row  = rkey & 0x3FFF;
        uint8_t  bank = (rkey >> 14) & 0x3;
        uint8_t  bg   = (rkey >> 16) & 0x3;
        uint8_t  pc   = (rkey >> 18) & 0x1;
        const uint8_t CH = 0;          // broadcast fans out to the enabled channels

        prog.add_command(SoftMCPlatform::CMD_ACT, row, 0, bg, bank, pc, CH);
        prog.add_nops(20);             // tRCD
        for (uint8_t col : kv.second) {
            prog.add_command(SoftMCPlatform::CMD_WR, row, col, bg, bank, pc, CH);
            uint8_t wd[32];
            if (randomize) {
                uint64_t akey = ((uint64_t)rkey << 5) | col;
                uint64_t s = ((uint64_t)seed * 0x9E3779B97F4A7C15ull) ^ (akey + 0x123456789ABCDEFull);
                for (int i = 0; i < 4; i++) { uint64_t v = splitmix64(s); memcpy(wd + i*8, &v, 8); }
            } else {
                memset(wd, 0, sizeof(wd));
            }
            prog.set_write_data(prog.size() - 1, wd);
            prog.add_nops(8);          // tCCD between writes
            total_written++;
        }
        prog.add_nops(20);             // tWR before precharge
        prog.add_command(SoftMCPlatform::CMD_PRE, row, 0, bg, bank, pc, CH);
        prog.add_nops(20);             // tRP

        if (prog.size() >= FLUSH_CMDS) { if (flush()) return 1; }
    }
    if (flush()) return 1;

    cout << "Init-data: wrote " << total_written << " words." << endl;
    return 0;
}

int main(int argc, char* argv[])
{
    string   csv_path      = DEFAULT_TEST_PATH;
    uint32_t iterations    = 0;            // 0 = infinite (until Ctrl+C)
    string   channels_mode = "all";
    uint64_t period_ms     = 1000;
    string   output_path;
    bool     garbage_reads = true;
    uint32_t wr_pattern    = 0xDEADBEEF;
    uint64_t tail_gap      = 64;           // trailing idle slots for tRP at the loop seam
    bool     verify        = false;
    bool     verify_stream = false;
    string   init_mode     = "none";      // none | zeros | random (pre-write footprint)
    uint32_t init_seed     = 1;           // seed for --init random

    enum { OPT_CSV = 256, OPT_ITER, OPT_CHANNELS, OPT_PERIOD, OPT_OUTPUT,
           OPT_NO_GR, OPT_WR_PATTERN, OPT_TAIL_GAP, OPT_VERIFY, OPT_VERIFY_STREAM,
           OPT_INIT, OPT_INIT_SEED };
    static struct option long_opts[] = {
        {"help",            no_argument,       nullptr, 'h'},
        {"csv",             required_argument, nullptr, OPT_CSV},
        {"iterations",      required_argument, nullptr, OPT_ITER},
        {"channels",        required_argument, nullptr, OPT_CHANNELS},
        {"period-ms",       required_argument, nullptr, OPT_PERIOD},
        {"output",          required_argument, nullptr, OPT_OUTPUT},
        {"no-garbage-reads",no_argument,       nullptr, OPT_NO_GR},
        {"wr-pattern",      required_argument, nullptr, OPT_WR_PATTERN},
        {"tail-gap",        required_argument, nullptr, OPT_TAIL_GAP},
        {"verify",          no_argument,       nullptr, OPT_VERIFY},
        {"verify-stream",   no_argument,       nullptr, OPT_VERIFY_STREAM},
        {"init",            required_argument, nullptr, OPT_INIT},
        {"init-seed",       required_argument, nullptr, OPT_INIT_SEED},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'h':            printUsage(argv[0]); return 0;
        case OPT_CSV:        csv_path      = optarg; break;
        case OPT_ITER:       iterations    = (uint32_t)strtoul(optarg, nullptr, 0); break;
        case OPT_CHANNELS:   channels_mode = optarg; break;
        case OPT_PERIOD:     period_ms     = (uint64_t)strtoull(optarg, nullptr, 0); break;
        case OPT_OUTPUT:     output_path   = optarg; break;
        case OPT_NO_GR:      garbage_reads = false; break;
        case OPT_WR_PATTERN: wr_pattern    = (uint32_t)strtoul(optarg, nullptr, 16); break;
        case OPT_TAIL_GAP:   tail_gap      = (uint64_t)strtoull(optarg, nullptr, 0); break;
        case OPT_VERIFY:     verify        = true; break;
        case OPT_VERIFY_STREAM: verify_stream = true; break;
        case OPT_INIT:       init_mode     = optarg; break;
        case OPT_INIT_SEED:  init_seed     = (uint32_t)strtoul(optarg, nullptr, 0); break;
        default:             printUsage(argv[0]); return 1;
        }
    }

    vector<int> channels = parse_channels(channels_mode);
    if (channels.empty()) {
        cerr << "ERROR: no valid channels parsed from --channels '" << channels_mode << "'" << endl;
        return 1;
    }

    if (init_mode != "none" && init_mode != "zeros" && init_mode != "random") {
        cerr << "ERROR: --init must be 'none', 'zeros', or 'random' (got '" << init_mode << "')" << endl;
        return 1;
    }

    // ---------------------------------------------------------------
    // Self-test mode: no CSV; write a pattern and read it back.
    // --verify-stream uses the PCIe stream path; --verify uses bram_replay.
    // Pass both to compare them on the same bitstream.
    // ---------------------------------------------------------------
    if (verify || verify_stream) {
        SoftMCPlatform platform;
        g_platform = &platform;
        int rc = platform.init();
        if (rc != SOFTMC_SUCCESS) {
            cerr << "ERROR: platform.init() failed (rc=" << rc << ")" << endl;
            return 1;
        }
        signal(SIGINT, sigint_handler);
        platform.reset_fpga();
        int result = 0;
        if (verify_stream) result |= run_verify_stream(platform, channels[0]);
        if (verify)        result |= run_verify(platform, channels[0]);
        return result;
    }

    // ---------------------------------------------------------------
    // Load + compress the CSV trace
    // ---------------------------------------------------------------
    cout << "Loading trace: " << csv_path << endl;
    vector<StreamProgram::TraceEntry> entries;
    uint64_t max_ts = 0;
    if (!StreamProgram::load_csv_entries(csv_path, entries, &max_ts)) {
        cerr << "ERROR: failed to load/compress CSV trace '" << csv_path << "'" << endl;
        return 1;
    }

    // Trace length (one iteration) = highest slot index + 1, rounded up to a
    // multiple of 4 (the engine advances 4 slots per fab cycle).
    // Append a trailing idle gap so the loop seam leaves >= tRP between the
    // closing PREs and the next iteration's re-opening ACTs. The engine fills
    // these extra slots with NOPs before wrapping to the next iteration.
    uint64_t trace_len = max_ts + 1 + tail_gap;
    if (trace_len % 4 != 0) trace_len += (4 - (trace_len % 4));

    // Split into parallel arrays + per-bank counts (bank = timestamp mod 4)
    vector<uint64_t> timestamps(entries.size());
    vector<uint32_t> commands(entries.size());
    size_t bank_count[4] = {0,0,0,0};
    bool   has_writes = false;
    for (size_t i = 0; i < entries.size(); i++) {
        timestamps[i] = entries[i].timestamp;
        commands[i]   = entries[i].command;
        bank_count[entries[i].timestamp & 3]++;
        uint8_t ct = (commands[i] >> 28) & 0xF;
        if (ct == SoftMCPlatform::CMD_WR || ct == SoftMCPlatform::CMD_WRA) has_writes = true;
    }

    cout << "  Compressed trace: " << entries.size() << " non-NOP commands, "
         << "trace length " << trace_len << " slots"
         << " (incl. " << tail_gap << " trailing idle for tRP)" << endl;
    cout << "  Per-bank counts (ts mod 4): "
         << bank_count[0] << " / " << bank_count[1] << " / "
         << bank_count[2] << " / " << bank_count[3]
         << "  (bank capacity " << BANK_DEPTH << ")" << endl;

    size_t max_bank = 0;
    for (int b = 0; b < 4; b++) if (bank_count[b] > max_bank) max_bank = bank_count[b];
    if (max_bank > BANK_DEPTH) {
        cerr << "ERROR: a bank needs " << max_bank << " entries but BANK_DEPTH="
             << BANK_DEPTH << ".\n       Increase BANK_DEPTH in bram_replay.v (and this app) and rebuild."
             << endl;
        return 1;
    }
    if (trace_len > 0xFFFFFFFFull) {
        cerr << "ERROR: trace length " << trace_len << " exceeds the 32-bit slot field." << endl;
        return 1;
    }

    // ---------------------------------------------------------------
    // Platform bring-up
    // ---------------------------------------------------------------
    SoftMCPlatform platform;
    g_platform = &platform;

    int rc = platform.init();
    if (rc != SOFTMC_SUCCESS) {
        cerr << "ERROR: platform.init() failed (rc=" << rc << ")" << endl;
        return 1;
    }
    signal(SIGINT, sigint_handler);

    platform.reset_fpga();
    platform.set_broadcast_channels(channels);

    // Optional one-time data init over the stream path (zeros or per-word
    // random), before the measured loop. Done with garbage-reads OFF (it's a
    // write-only pass) and no reset afterwards, so the data is still resident
    // when the BRAM loop starts reading it.
    if (init_mode != "none") {
        platform.set_garbage_reads(false);
        if (run_init_data(platform, entries, init_mode == "random", init_seed) != 0) {
            cerr << "ERROR: data init (" << init_mode << ") failed" << endl;
            platform.reset_fpga();
            return 1;
        }
    }

    if (garbage_reads) platform.set_garbage_reads(true);   // drop RD data so the loop never backs up C2H
    platform.initializeMonitoring();

    if (output_path.empty()) {
        mkdir("results", 0755);
        // Tag the file with the data-init scheme, so different data runs of the
        // same trace land in distinct CSVs: per-address random (rand<seed>) or
        // the single 8-hex write pattern.
        char patbuf[24];
        if      (init_mode == "random") snprintf(patbuf, sizeof(patbuf), "rand%u", init_seed);
        else if (init_mode == "zeros")  snprintf(patbuf, sizeof(patbuf), "zeros");
        else                            snprintf(patbuf, sizeof(patbuf), "%08x", wr_pattern);
        output_path = "results/" + derive_trace_name(csv_path)
                    + "_data_" + patbuf + "_bram.csv";
    } else {
        size_t slash = output_path.find_last_of('/');
        if (slash != string::npos) mkdir(output_path.substr(0, slash).c_str(), 0755);
    }
    platform.setCSVFilePath(output_path);
    platform.resetMonitoring();
    cout << "Metrics CSV: " << output_path << endl;
    cout << "Garbage reads: " << (garbage_reads ? "ON" : "OFF")
         << "   Iterations: " << (iterations ? to_string(iterations) : string("infinite (Ctrl+C to stop)"))
         << endl;

    // ---------------------------------------------------------------
    // Watcher thread: turns a Ctrl+C (g_stop_requested) into a graceful
    // stopTraceLoop() on the H2C channel while the main thread blocks on C2H.
    // ---------------------------------------------------------------
    std::atomic<bool> watcher_run{true};
    std::thread watcher([&]() {
        bool sent = false;
        while (watcher_run.load()) {
            if (!sent && g_stop_requested.load()) {
                platform.stopTraceLoop();
                sent = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    // ---------------------------------------------------------------
    // HBM monitoring thread
    // ---------------------------------------------------------------
    g_monitoring_running = true;
    g_monitor_thread = std::thread([&platform, period_ms]() {
        while (g_monitoring_running.load()) {
            platform.readAndPrintHBMMetrics();
            auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(period_ms);
            while (g_monitoring_running.load() && std::chrono::steady_clock::now() < end)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // ---------------------------------------------------------------
    // Load into BRAM, start the loop, wait for completion
    // ---------------------------------------------------------------
    cout << "Loading trace into BRAM and starting replay..." << endl;
    auto wall_start = std::chrono::steady_clock::now();

    if (platform.loadTraceAndRun(timestamps.data(), commands.data(), entries.size(),
                                 trace_len, iterations, wr_pattern, has_writes) != 0) {
        cerr << "ERROR: loadTraceAndRun failed" << endl;
        watcher_run = false; if (watcher.joinable()) watcher.join();
        stop_monitoring();
        platform.reset_fpga();
        return 1;
    }

    uint64_t fpga_cycles = platform.waitTraceDone();
    auto wall_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    watcher_run = false;
    if (watcher.joinable()) watcher.join();
    stop_monitoring();
    platform.reset_fpga();

    // ---------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------
    // Each fab cycle issues 4 slots; one iteration is ceil(trace_len/4) fab
    // cycles plus a fixed re-prime overhead between iterations.
    double fpga_s          = (double)fpga_cycles / (FAB_CLK_MHZ * 1e6);
    uint64_t cyc_per_iter  = (trace_len + 3) / 4;       // ideal (no prime overhead)
    uint64_t iters_done    = (iterations != 0) ? iterations
                            : (cyc_per_iter ? (fpga_cycles / cyc_per_iter) : 0);

    // Command-stream throughput: 4 B per slot, 4 slots/fab cycle (peak 2.4 GB/s).
    // BRAM replay has no DMA stalls, so sustained ~= peak.
    const double CMD_BYTES      = 4.0;
    double total_slots          = (double)fpga_cycles * 4.0;
    double slot_throughput_GBs  = fpga_s > 0 ? total_slots * CMD_BYTES / fpga_s / 1e9 : 0.0;
    double useful_cmds          = (double)entries.size() * (double)iters_done;
    double useful_GBs           = fpga_s > 0 ? useful_cmds * CMD_BYTES / fpga_s / 1e9 : 0.0;

    cout << "\n=== Results (BRAM replay) ===" << endl;
    cout << "Iterations " << (iterations ? "(requested): " : "(completed, est.): ") << iters_done << endl;
    cout << "Total FPGA cycles:    " << fpga_cycles
         << " (" << fixed << setprecision(6) << fpga_s << " s @150 MHz)" << endl;
    cout << "Wall time:            " << setprecision(6) << wall_s << " s" << endl;
    cout << "Non-NOP commands:     " << (uint64_t)useful_cmds << endl;
    cout << "\n--- Command-stream throughput (4 B/slot) ---" << endl;
    cout << "Slot rate (sustained):" << setprecision(3) << slot_throughput_GBs
         << " GB/s  (peak 2.400 GB/s; BRAM replay has no DMA stalls)" << endl;
    cout << "Useful command rate:  " << setprecision(3) << useful_GBs << " GB/s" << endl;
    cout << "Metrics CSV:          " << output_path << endl;

    return 0;
}
