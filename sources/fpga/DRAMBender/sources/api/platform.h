#include "board.h"
#include "prog.h"
#include <thread>
#include <string>
#include <vector>
#include <cstdint>
#include <boost/lockfree/spsc_queue.hpp>

class StreamProgram;  // forward declaration

#ifdef PYSMC
#include "ext/pybind11/include/pybind11/pybind11.h"
namespace py = pybind11;
#endif

//ERROR CODES
#define SOFTMC_SUCCESS 0
#define SOFTMC_ERR -1
#define SOFTMC_NO_PLATFORM -2
#define SOFTMC_ERR_OPEN_FPGA -3
#define SOFTMC_NO_SUCH_FPGA -4

class SoftMCPlatform{
  #define INSTR_BUF_SIZE 32*2048*64
  #define API_BUF_SIZE 1024*1024*2
  #define DEFAULT_CSV_PATH "hbm_metrics.csv"
  public:
    SoftMCPlatform();
    SoftMCPlatform(bool);
    SoftMCPlatform(int dimm_select, bool garbage_reads_enabled = false);
    ~SoftMCPlatform();
    /**
     * Initializes the whole platform
     * @return SOFTMC_SUCCESS on sucessful initialization
     */
    int init();
    /**
     * Resets SoftMC logic, won't reset PCI-E endpoint or the PHY interface
     */
    void reset_fpga();
    /**
     * Sends SoftMC program to the FPGA board over PCI-E
     * @param prog reference to the program object to send
     */
    void execute(Program & prog);
    /**
     * Receive data from the FPGA board over PCI-E
     * @param dst_buf pointer to the buffer that will receive the data
     * @param num_words number of bytes to read
     */
    int receiveData(void* dst_buf, int num_words);

    #ifdef PYSMC
    int py_receiveData(int num_words);
    #endif

    /**
     * Compare data with a given data pattern (repeating bytes)
     * and return number of bitflips in 8KB of data
     * @param comp_pattern one byte data pattern to compare the read data
     */ 
    int count_bitflips_in_row(unsigned char comp_pattern);

    /**
     * Turn auto-refresh on-off
     * @param on true to turn aref on false otherwise
     */
    void set_aref(bool on);

    void read_HBM_temperature();
    int return_HBM_temperature();

    void set_broadcast_channels(std::vector<int>);
    void set_garbage_reads(bool);

    // ---------------------------------------------------------------
    // 32-bit command encoding utilities
    // ---------------------------------------------------------------

    /**
     * Encode a single HBM command into the 32-bit stream format.
     *
     * Bit layout:
     *   [31:28] CMD_TYPE (4 bits) — project.vh encoding
     *   [27:14] ROW      (14 bits)
     *   [13:9]  COL      (5 bits)
     *   [8:7]   BG       (2 bits)
     *   [6:5]   BANK     (2 bits)
     *   [4]     PC       (1 bit)  — pseudo-channel
     *   [3:0]   CH       (4 bits) — HBM channel
     */
    static uint32_t encodeCommand(uint8_t cmd_type, uint16_t row, uint8_t col,
                                  uint8_t bg, uint8_t bank, uint8_t pc, uint8_t ch);

    /** Encode a NOP command (cmd_type = 0). */
    static uint32_t encodeNOP();

    // Command type constants (matching project.vh)
    static const uint8_t CMD_NOP  = 0;
    static const uint8_t CMD_ACT  = 1;
    static const uint8_t CMD_PRE  = 2;
    static const uint8_t CMD_PREA = 3;
    static const uint8_t CMD_REF  = 5;
    static const uint8_t CMD_RD   = 10;
    static const uint8_t CMD_RDA  = 11;
    static const uint8_t CMD_WR   = 12;
    static const uint8_t CMD_WRA  = 13;

    // ---------------------------------------------------------------
    // Stream execution (sends commands to HBM, receives read data)
    // ---------------------------------------------------------------

    /**
     * Prepare a stream command buffer from a vector of 32-bit encoded commands.
     * Pads to 32-byte alignment and appends end marker.
     *
     * @param commands       vector of encoded commands
     * @param out_total_bytes [out] total byte size of returned buffer
     * @return page-aligned buffer (caller must free())
     */
    void* prepareStreamCommands(const std::vector<uint32_t>& commands, size_t* out_total_bytes);

    /**
     * Prepare a stream command buffer from a StreamProgram, interleaving
     * write-data beats after each 8-command group that contains WR/WRA
     * commands.  When the StreamProgram carries write data (per-command
     * or default), the header flag wr_data_enable is set so that the
     * FPGA splitter routes data beats to the write-data FIFO.
     *
     * @param prog             StreamProgram with commands and optional write data
     * @param out_total_bytes  [out] total byte size of returned buffer
     * @return page-aligned buffer (caller must free())
     */
    void* prepareStreamCommands(const StreamProgram& prog, size_t* out_total_bytes);

    /**
     * Execute a stream of HBM commands and collect results.
     *
     * Sends the sideband to enter stream mode, streams the buffer, then
     * receives any HBM read data and the final cycle count.
     *
     * @param data         prepared buffer from prepareStreamCommands()
     * @param size_bytes   total buffer size
     * @param read_data    [out] received HBM read data (512 bits per read, as 64 bytes)
     * @param cycle_count  [out] FPGA cycle count
     * @return 0 on success, nonzero on error
     */
    int streamExecute(const void* data, size_t size_bytes,
                      std::vector<uint8_t>& read_data, uint64_t& cycle_count);

    /**
     * Legacy: run a stream throughput test (no real commands, no read data).
     * Kept for backward compatibility.
     */
    uint64_t streamTest(const void* data, size_t size_bytes, uint32_t loop_count = 1);

    /**
     * Legacy: prepare a stream throughput test buffer with divider header.
     */
    void* prepareStreamData(const void* payload, size_t payload_bytes, size_t* out_total_bytes, uint32_t consume_divider = 1);

    // ---------------------------------------------------------------
    // BRAM trace-replay control (on-chip looped replay, no PCIe during run)
    // ---------------------------------------------------------------

    /** Enter BRAM replay mode, load a compressed command trace into the on-chip
     *  replay memory, configure the loop, and arm the engine (it begins replaying
     *  as soon as the load completes). Pair with waitTraceDone() to block for the
     *  result and stopTraceLoop() to stop early.
     *  @param timestamps         absolute slot index per entry [n_entries]
     *  @param commands           32-bit encoded command per entry [n_entries]
     *  @param n_entries          number of compressed (non-NOP) entries
     *  @param trace_length_slots total slots in one iteration (>= max ts + 1)
     *  @param num_iterations     loop count (0 = run until stopTraceLoop())
     *  @param wr_pattern         32-bit write pattern for WR/WRA (replicated x16)
     *  @param wr_data_enable     use wr_pattern (else a built-in dummy pattern)
     *  @return 0 on success, nonzero on error */
    int loadTraceAndRun(const uint64_t* timestamps, const uint32_t* commands,
                        size_t n_entries, uint64_t trace_length_slots,
                        uint32_t num_iterations, uint32_t wr_pattern = 0xDEADBEEF,
                        bool wr_data_enable = false);

    /** Block until the replay engine completes (all iterations done, or a
     *  stopTraceLoop() request was honoured at an iteration boundary). Returns
     *  the total replay duration in FPGA (fab_clk) cycles. */
    uint64_t waitTraceDone();

    /** Like waitTraceDone(), but also captures the HBM read data returned over
     *  C2H (requires garbage reads OFF). @p read_data receives the read bytes.
     *  If @p strip_cycle_count is true the {8 x cycle_count} flush block is
     *  detected and removed (heuristic: 8 equal non-zero 64-bit lanes — note
     *  this can mis-fire on uniform read data); if false, ALL returned entries
     *  are kept verbatim. Returns the FPGA cycle count (0 if not stripping). */
    uint64_t waitTraceDoneCapture(std::vector<uint8_t>& read_data,
                                  bool strip_cycle_count = true);

    /** Request a graceful stop: the engine finishes the current iteration then
     *  idles. Safe to call from a signal handler / another thread while
     *  waitTraceDone() is blocked (uses the H2C channel; waitTraceDone uses C2H). */
    void stopTraceLoop();

    /**
     * Used along with Program::dumpRegisters to read register content
     */
    void readRegisterDump();

    //Modifying the dimm_select to access different FPGA in the cluster.
    void setDimmSelect(int new_dimm_select); 	

    // Verifying the new dimm_select. 
    int getDimmSelect() const;

    void blind_receive();

    /**
     * Initialize the HBM monitoring system
     * @return SOFTMC_SUCCESS on successful initialization, SOFTMC_ERR otherwise
     */
    int initializeMonitoring();

    /**
     * Reset the HBM monitoring registers
     */
    void resetMonitoring();

    /**
     * Read all HBM monitoring metrics and print them to stdout and append to CSV
     * Prints: voltage (INS/MAX/AVG) and current (INS/MAX/AVG)
     */
    void readAndPrintHBMMetrics();

    /**
     * Get the average current from HBM monitoring registers (proxy for power)
     * @return average current in mA
     */
    uint32_t getAverageCurrent();

    /**
     * Set the CSV file path for HBM metrics
     * @param path path to the CSV file
     */
    void setCSVFilePath(const std::string& path);

    /**
     * Get the current CSV file path for HBM metrics
     * @return current CSV file path
     */
    std::string getCSVFilePath() const;

    /**
     * Start a background thread that records HBM metrics at a specified interval
     * and writes them to a CSV file
     * @param sleep_ms time in milliseconds between metric recordings
     */
    void startMetricsThread(uint64_t sleep_ms);

    /**
     * Start a background thread that simply records HBM metrics for a fixed
     * duration (no stabilization checks).  Resets the FPGA when done.
     * @param sleep_ms time in milliseconds between metric recordings
     * @param duration_s total recording duration in seconds (default: 3600 = 1 hour)
     */
    void startMetricsThreadFixedDuration(uint64_t sleep_ms, int duration_s = DEFAULT_FIXED_DURATION_S);

    /**
     * Stop the background metrics thread
     */
    void stopMetricsThread();

    /**
     * Check if the metrics thread is running
     * @return true if metrics thread is running, false otherwise
     */
    bool metricsThreadRunning() const { return metrics_thread_running; }

    bool receiverRunning();

    /**
     * Start a background temperature watchdog thread.
     * Requires initializeMonitoring() to have been called first.
     * The watchdog reads the HBM temperature at the given interval
     * and resets the FPGA + stops execution when the temperature
     * exceeds the configured maximum.
     * @param check_interval_ms polling interval in milliseconds (default: 1000)
     */
    void startTemperatureWatchdog(uint64_t check_interval_ms = 1000);

    /**
     * Stop the temperature watchdog thread
     */
    void stopTemperatureWatchdog();

    /**
     * Set the maximum allowed HBM temperature in degrees Celsius.
     * When this temperature is exceeded the watchdog resets the FPGA.
     * @param temp maximum temperature in Celsius (default: 93)
     */
    void setMaxTemperature(int temp);

    /**
     * Get the configured maximum temperature threshold
     * @return maximum temperature in Celsius
     */
    int getMaxTemperature() const;

    /**
     * Check if the temperature watchdog is running
     * @return true if watchdog is running, false otherwise
     */
    bool temperatureWatchdogRunning() const { return temp_watchdog_running; }

    /**
     * Check if an over-temperature shutdown has occurred
     * @return true if the FPGA was reset due to over-temperature
     */
    bool wasOverTemperatureShutdown() const { return over_temp_shutdown; }

  private:
    bool is_dummy;
    int dimm_select;
    int execute_counter;
    int read_bytes_total = 0;
    bool garbage_reads_enabled = false;
    bool receiver_done = false;
    bool receiver_should_stop = false;

    BoardInterface *iface;
    void* instr_buf;
    std::thread receiver;
    boost::lockfree::spsc_queue<int, boost::lockfree::capacity<API_BUF_SIZE/4>> api_recv_buf;
    void* xdma_recv_buf;
    void consumeData();

    // HBM monitoring members
    void* monitor_map_base = nullptr;
    int monitor_fd = -1;
    size_t monitor_map_size = 0;
    bool monitoring_initialized = false;

    // CSV logging members
    std::string csv_file_path = DEFAULT_CSV_PATH;
    bool csv_header_written = false;

    // Metrics thread members
    std::thread metrics_thread;
    bool metrics_thread_running = false;
    
    // Power stabilization tracking
    // Phase 1a: coarse stabilization (smaller window, larger range)
    static const int COARSE_STABILIZATION_WINDOW = 10;
    static const uint32_t COARSE_STABILIZATION_RANGE_MA = 30;
    // Phase 1b: fine stabilization (longer window, narrower range)
    static const int FINE_STABILIZATION_WINDOW = 1;
    static const uint32_t FINE_STABILIZATION_RANGE_MA = 100;
    // Phase 2: data collection duration
    static const int COLLECTION_DURATION_S = 50;
    // Reset monitoring registers every N measurements
    static const int METRICS_RESET_INTERVAL = 40;
    // Default fixed-duration mode duration (1 hour)
    static const int DEFAULT_FIXED_DURATION_S = 3600;

    // Temperature watchdog members
    std::thread temp_watchdog_thread;
    bool temp_watchdog_running = false;
    bool over_temp_shutdown = false;
    int max_temperature = 93;  // default threshold in Celsius

  #ifdef PYSMC
  public:
    uint8_t* py_data_buffer = nullptr;
    py::memoryview get_buffer_memoryview();
  #endif
};