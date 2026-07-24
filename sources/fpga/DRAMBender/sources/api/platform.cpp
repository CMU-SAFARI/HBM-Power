#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cassert>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <boost/lockfree/spsc_queue.hpp>

#include "platform.h"
#include "board.h"
#include "prog.h"
#include "stream_prog.h"

/* HBM Monitoring Functions */

#define PAGE_SIZE sysconf(_SC_PAGESIZE)

/* Register addresses */
#define CONTROL_REG_ADDR       0x028018
#define HOST_STATUS2_REG_ADDR  0x02830C
#define MB_RESETN_REG          0x020000
#define ERROR_REG              0x02800C
#define PROFILE_NAME_REG       0x028014
#define FW_VERSION_REG         0x028004
#define STATUS_REG             0x028008

#define HBM_TEMP1_INS_REG      0x028268
#define HBM_TEMP1_MAX_REG      0x028260
#define HBM_TEMP1_AVG_REG      0x028264

#define HBM_TEMP2_INS_REG      0x0282BC
#define HBM_TEMP2_MAX_REG      0x0282B4
#define HBM_TEMP2_AVG_REG      0x0282B8

#define HBM_1V2_INS_REG        0x028298
#define HBM_1V2_MAX_REG        0x028290
#define HBM_1V2_AVG_REG        0x028294

#define HBM_1V2_I_INS_REG      0x028418
#define HBM_1V2_I_MAX_REG      0x028410
#define HBM_1V2_I_AVG_REG      0x028414

#define PEX_12V_INS_REG        0x028028
#define PEX_12V_MAX_REG        0x028020
#define PEX_12V_AVG_REG        0x028024

#define I_12VPEX_IN_INS_REG        0x0280d0
#define I_12VPEX_IN_MAX_REG        0x0280c8
#define I_12VPEX_IN_AVG_REG        0x0280cc

#define PEX_3V3_INS_REG        0x028034
#define PEX_3V3_MAX_REG        0x02802c
#define PEX_3V3_AVG_REG        0x028030

#define I_3V3PEX_IN_INS_REG        0x028280
#define I_3V3PEX_IN_MAX_REG        0x028278
#define I_3V3PEX_IN_AVG_REG        0x02827c

#define VCCINT_INS_REG        0x0280e8
#define VCCINT_MAX_REG        0x0280e0
#define VCCINT_AVG_REG        0x0280e4

#define VCCINT_I_INS_REG        0x0280f4
#define VCCINT_I_MAX_REG        0x0280ec
#define VCCINT_I_AVG_REG        0x0280f0

#define VCCINT_IO_INS_REG        0x0282b0
#define VCCINT_IO_MAX_REG        0x0282a8
#define VCCINT_IO_AVG_REG        0x0282ac

#define VCCINT_IO_I_INS_REG        0x02828c
#define VCCINT_IO_I_MAX_REG        0x028284
#define VCCINT_IO_I_AVG_REG        0x028288

[[maybe_unused]] static int consume_total = 0;
[[maybe_unused]] static int receive_total = 0;

SoftMCPlatform::SoftMCPlatform()
{
  // 32 KB large read buffer
  is_dummy = false;
  xdma_recv_buf = malloc(128*1024);
  this->dimm_select = 0; // by default, use DIMM 0 in C3 (or A in the diagram on the spreadsheet)
  this->execute_counter = 0;
  receiver_should_stop = false;
  #ifdef PYSMC
  py_data_buffer = (uint8_t*)malloc(32*1024*sizeof(uint8_t));
  #endif
}

SoftMCPlatform::SoftMCPlatform(bool sandbox)
{
  // 32 KB large read buffer
  is_dummy = sandbox;
  xdma_recv_buf = malloc(128*1024);
  this->dimm_select = 0;
  this->execute_counter = 0;
  receiver_should_stop = false;

}

SoftMCPlatform::SoftMCPlatform(int dimm_select, bool garbage_reads_enabled)
{
  // 32 KB large read buffer
  is_dummy = false;
  xdma_recv_buf = malloc(128*1024);

  this->dimm_select = dimm_select;
  this->execute_counter = 0;
  this->garbage_reads_enabled = garbage_reads_enabled;
  receiver_should_stop = false;

  #ifdef PYSMC
  py_data_buffer = (uint8_t*)malloc(32*1024*sizeof(uint8_t));
  #endif
}

SoftMCPlatform::~SoftMCPlatform(){
  // Stop temperature watchdog if running
  if (temp_watchdog_running) {
    stopTemperatureWatchdog();
  }

  // Stop metrics thread if running
  if (metrics_thread_running) {
    stopMetricsThread();
  }

  if (receiver.joinable())
    receiver.join();

  if (xdma_recv_buf)
    free(xdma_recv_buf);

  if (instr_buf)
    free(instr_buf);

  if (iface)
    delete iface;

  // Clean up monitoring resources
  if (monitor_map_base != nullptr && monitor_map_size > 0) {
    munmap(monitor_map_base, monitor_map_size);
  }

  if (monitor_fd >= 0) {
    close(monitor_fd);
  }

  #ifdef PYSMC
  if (py_data_buffer)
    free(py_data_buffer);
  #endif
}

int SoftMCPlatform::init(){
  if(is_dummy)
  {
    instr_buf = malloc(INSTR_BUF_SIZE);
    memset(instr_buf, 0, INSTR_BUF_SIZE);
    return SOFTMC_SUCCESS;
  }
  else
  {
    instr_buf = malloc(INSTR_BUF_SIZE);
    memset(instr_buf, 0, INSTR_BUF_SIZE);

    iface = new BoardInterface(BoardInterface::IFACE::XDMA, this->dimm_select, garbage_reads_enabled);
  
    int init = iface -> init();

    if (init)
      return SOFTMC_ERR;

    return SOFTMC_SUCCESS;
  }
}

/**
 * This sends a 256 bit data which has it's
 * 33rd bit set to '1'.
 */
void SoftMCPlatform::reset_fpga()
{
  if(is_dummy)
  {
    return;
  }
  else
  {
    // join receiver thread
    if(receiver.joinable())
      receiver.join();

    ((uint8_t*) instr_buf)[8] = (uint8_t) 1;
    int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
    // We do not need to zero out the whole buffer
    memset(instr_buf, 0, 32);
    if(sent)
      std::cerr << "Could not reset the FPGA!" << std::endl;
    else
      std::cout << "Successfully reset the FPGA!" << std::endl;

    // sleep (1);

    // blind_receive();

    // api_recv_buf.consume_all([](int i) {});

    // sleep(1);
  }
}

void SoftMCPlatform::blind_receive()
{
  receiver = std::thread(&SoftMCPlatform::consumeData, this);

  sleep(1);

  if(receiver.joinable())
    receiver.join();
}

void SoftMCPlatform::execute(Program &prog)
{
  if(is_dummy)
  {
    [[maybe_unused]] uint64_t* iseq     = (uint64_t*) prog.get_inst_array();
    int bytes          = prog.size();
    assert (bytes <= INSTR_BUF_SIZE/4 && " too many instructions in the buffer, the limit is 2048.");
    return;
  }
  else
  {
    // Temperature safeguard: refuse to execute if over-temperature shutdown occurred
    if (over_temp_shutdown) {
      std::cerr << "Refusing to execute: FPGA was reset due to over-temperature. "
                << "Resolve the thermal issue before continuing." << std::endl;
      free(prog.get_inst_array());
      return;
    }

    // Temperature safeguard: check current temperature before executing (if monitoring active)
    if (monitoring_initialized) {
      off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);
      volatile uint32_t *temp1_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP1_INS_REG - map_phys_base));
      volatile uint32_t *temp2_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP2_INS_REG - map_phys_base));
      uint32_t t1 = *temp1_ins;
      uint32_t t2 = *temp2_ins;
      uint32_t t_max = std::max(t1, t2);

      if ((int)t_max >= max_temperature) {
        std::cerr << "Refusing to execute: HBM temperature " << t_max
                  << " °C exceeds threshold " << max_temperature << " °C. "
                  << "Resetting FPGA." << std::endl;
        over_temp_shutdown = true;
        receiver_should_stop = true;
        if (garbage_reads_enabled && iface != nullptr) {
          iface->disableGarbageReads();
          iface->closeReceiveDescriptor();
        }
        reset_fpga();
        free(prog.get_inst_array());
        return;
      }
    }

    uint64_t* iseq     = (uint64_t*) prog.get_inst_array();
    uint64_t* temp_ptr = (uint64_t*) instr_buf;
    int bytes          = prog.size();
    assert (bytes <= INSTR_BUF_SIZE/4 && " too many instructions in the buffer, the limit is 2048.");

    for(int i = 0 ; i < bytes/8 ; i++)
    {
      temp_ptr[i*4] = iseq[i];
      // Set the 69th bit on the last temp_ptr element to indicate end of program
      if(i == bytes/8 - 1)
        ((uint8_t*) &(temp_ptr[i*4]))[8] = (uint8_t) 0x20;
    }

    if(receiver.joinable())
      receiver.join();

    int sent = iface -> sendData(instr_buf, bytes*4 /*in bytes*/);

    memset(instr_buf, 0, bytes*4);
    free(iseq);
    assert(!sent && "could not send instructions");

    // print the number of executions
    std::cout << "Executed " << ++execute_counter << " programs" << std::endl;
    // print number of bytes read
    std::cout << "Read " << read_bytes_total << " bytes" << std::endl;

    
    if(!garbage_reads_enabled) {
      receiver = std::thread(&SoftMCPlatform::consumeData, this);
    }
  }
}

bool SoftMCPlatform::receiverRunning()
{
  return (receiver.joinable() && !receiver_done);
}

void SoftMCPlatform::consumeData()
{
  receiver_done = false;
  while(!receiver_should_stop)
  {
    int read_meta_size = 32;

    int recvd = iface -> recvData((void*)xdma_recv_buf, read_meta_size);

    read_bytes_total += recvd;

    // cast the first dword of xdma_recv_buf into a variable
    // that we can use to check the status of the transfer

    uint64_t * meta = (uint64_t *) xdma_recv_buf;
    // the first 12 bits is the size of the next transfer
    int xdma_read_size = (int) (meta[0] & 0x00000FFF);
    xdma_read_size = xdma_read_size * 32;
    // the last bit of the 4th dword is if this is the last transfer or not
    int is_last = ((meta[3] >> 63) & 1);

    // the next 32 bits to the right of "last transfer bit" is the transaction ID
    int transaction_id = (int) ((meta[3] >> 31) & 0x00000000FFFFFFFF);


    // printf("Received %d bytes\n", xdma_read_size);
    // printf("Transaction ID%d\n", transaction_id);
    // printf("is_last: %d\n", is_last);

    if (is_last && xdma_read_size == 0)
    {
      // printf("I would normally break here\n");
      break;
    }

    recvd = iface -> recvData((void*)xdma_recv_buf, xdma_read_size);

    read_bytes_total += recvd;

    xdma_read_size /= 4;
    int total_size = xdma_read_size;

    int pushsz = api_recv_buf.push(((int*) xdma_recv_buf), xdma_read_size);
    while(pushsz < total_size) {
      pushsz += api_recv_buf.push(((int*) xdma_recv_buf) + pushsz, xdma_read_size = (total_size - pushsz));
      if(receiver_should_stop) break;
    }

    assert(pushsz == total_size &&
      "Unexpected amount of data pushed into spsc\n");

    if (is_last)
    {
      // printf("I would normally break here\n");
      break;
    }
  }
  std::cout << "Receiver thread ending, receiver_should_stop: " << receiver_should_stop << std::endl;
  receiver_done = true;
}

/**
* Try to read param(size) bytes from FPGA, function will block until
* all data is read
* @param recv_buf where to copy read data
* @param size number of bytes to read
* returns the number of bytes read on success
*/
int SoftMCPlatform::receiveData(void* recv_buf, int size){
  if(is_dummy)
  {
    assert(size>0 && size%4 == 0 && "size is expected to be a multiple of four\n");
    size /= 4;
    int * my_buf = (int *) recv_buf;
    for(int i = 0; i < size ; ++i) {
      my_buf[i] = 0x0;
    }
    return size * 4;
  }
  else
  {
    assert(size>0 && size%4 == 0 && "size is expected to be a multiple of four\n");

    size /= 4;
    int total_size = size;
    int rdsz = api_recv_buf.pop((int*) recv_buf, size);
    while (rdsz < total_size)
      rdsz += api_recv_buf.pop(((int*) recv_buf) + rdsz, size = (total_size-rdsz));

    assert(rdsz == total_size && "Unexpected amount of data popped from spsc\n");
    return total_size*4;
  }
}

#ifdef PYSMC
int SoftMCPlatform::py_receiveData(int size){
  if (size > 32 * 1024)
  {
    std::cerr << "Python version only supports read buffer size of up to 32KB!" << std::endl;
    return 0;
  }

  if(is_dummy)
  {
    assert(size>0 && size%4 == 0 && "size is expected to be a multiple of four\n");
    size /= 4;
    int * my_buf = (int *) py_data_buffer;
    for(int i = 0; i < size ; ++i) {
      my_buf[i] = 0x0;
    }
    return size * 4;
  }
  else
  {
    assert(size>0 && size%4 == 0 && "size is expected to be a multiple of four\n");

    size /= 4;
    int total_size = size;
    int rdsz = api_recv_buf.pop((int*) py_data_buffer, size);
    while (rdsz < total_size)
      rdsz += api_recv_buf.pop(((int*) py_data_buffer) + rdsz, size = (total_size-rdsz));

    assert(rdsz == total_size && "Unexpected amount of data popped from spsc\n");
    return total_size*4;
  }
}

py::memoryview SoftMCPlatform::get_buffer_memoryview()
{
  return py::memoryview::from_memory((uint64_t*) py_data_buffer, sizeof(uint8_t) * 32 * 1024, true);
}
#endif

int SoftMCPlatform::count_bitflips_in_row(unsigned char comp_pattern){
  int num_bitflips = 0;
  unsigned char buf[8192];
  receiveData(buf, 8192); // read one row each iteration

  for(int j = 0 ; j < 8192 ; j++){        
    if(comp_pattern != buf[j])
    {
      for(int i = 0 ; i < 8 ; i++)
      {
        if(((comp_pattern >> i) & 1) != ((buf[j] >> i) & 1))
            num_bitflips ++;
      }
    }
  }

  return num_bitflips;
}

void SoftMCPlatform::set_aref(const bool on)
{
  if(is_dummy)
  {
    std::cout << (on ? "Enabled" : "Disabled") << " autorefresh!" << std::endl;
    return;
  }
  else
  {
    ((uint8_t*) instr_buf)[8] = (uint8_t) 0x8;
    ((uint8_t*) instr_buf)[0] = on;
    int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
    // We do not need to zero out the whole buffer
    memset(instr_buf, 0, 32);
    
    if(sent)
      std::cerr << "Could not set auto refresh!" << std::endl;
    // else
    //   std::cout << (on ? "Enabled" : "Disabled") << " autorefresh!" << std::endl;
  }
}

void SoftMCPlatform::read_HBM_temperature()
{
    ((uint8_t*) instr_buf)[8] = (uint8_t) 0x10;
    int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
    // We do not need to zero out the whole buffer
    memset(instr_buf, 0, 512);

    sent = iface -> sendData(instr_buf, 512 /*in bytes*/);

    int recvd = iface -> recvData((void*)xdma_recv_buf, 32*1024);

    printf("Temp1: %d Celsius\n", ((uint8_t*) xdma_recv_buf)[32]);
    printf("Temp2: %d Celsius\n", ((uint8_t*) xdma_recv_buf)[33]);
}

int SoftMCPlatform::return_HBM_temperature()
{
    ((uint8_t*) instr_buf)[8] = (uint8_t) 0x10;
    int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
    // We do not need to zero out the whole buffer
    memset(instr_buf, 0, 512);

    sent = iface -> sendData(instr_buf, 512 /*in bytes*/);

    int recvd = iface -> recvData((void*)xdma_recv_buf, 32*1024);

    // Return stack 2's temperature
    return (int)(((uint8_t*) xdma_recv_buf)[33]);
}

void SoftMCPlatform::set_broadcast_channels(std::vector<int> channels)
{
    unsigned int channel_enable_bitvector = 0;
    for (auto channel: channels){
        if (channel > 15) 
            std::cerr << "Cannot have more than 16 channels" << std::endl;
        channel_enable_bitvector |= (1U << channel);
    }


    if(is_dummy)
    {
        return;
    }
    else
    {
        ((uint8_t*) instr_buf)[8] = (uint8_t) 128;
        ((uint32_t*) instr_buf)[0] = channel_enable_bitvector;
        int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
        // We do not need to zero out the whole buffer
        memset(instr_buf, 0, 32);
        if(sent)
            std::cerr << "Could not set broadcast HBM channels!" << std::endl;
        else
            std::cout << "Successfully set broadcast HBM channels" << std::endl;
    }

}

void SoftMCPlatform::set_garbage_reads(bool enable)
{
    if(is_dummy)
    {
        return;
    }
    else
    {
        ((uint8_t*) instr_buf)[8] = (uint8_t) 64;
        ((uint32_t*) instr_buf)[0] = enable ? 0x1 : 0x0;
        int sent = iface -> sendData(instr_buf, 32 /*in bytes*/);
        // We do not need to zero out the whole buffer
        memset(instr_buf, 0, 32);
        if(sent)
            std::cerr << "Could not set garbage reads!" << std::endl;
        else
            std::cout << "Successfully set garbage reads" << std::endl;
    }

}

uint64_t SoftMCPlatform::streamTest(const void* data, size_t size_bytes, uint32_t loop_count)
{
    if(is_dummy)
    {
        std::cerr << "streamTest not available in dummy mode" << std::endl;
        return 0;
    }
    if(loop_count == 0) loop_count = 1;

    // --- Step 1: Send sideband command to enter stream test mode ---
    memset(instr_buf, 0, 32);
    ((uint8_t*) instr_buf)[9] = (uint8_t) 0x01;  // bit 72
    int sent = iface->sendData(instr_buf, 32);
    memset(instr_buf, 0, 32);
    if(sent) {
        std::cerr << "Could not enter stream test mode!" << std::endl;
        return 0;
    }
    std::cout << "Entered stream test mode" << std::endl;

    // --- Step 2: Send the stream data ---
    // Buffer layout from prepareStreamData():
    //   [HEADER_SIZE bytes config header]
    //   [payload ...]
    //   [end marker + padding (last 32 bytes)]
    //
    // For loop_count > 1, we send:
    //   1. The config header (once)
    //   2. The payload region (loop_count times)
    //   3. The end marker block (once)
    const size_t HEADER_SIZE = 32;  // must match prepareStreamData
    // The last 32 bytes contain the end marker + padding
    const size_t TRAILER_SIZE = 32;

    if(size_bytes < HEADER_SIZE + TRAILER_SIZE) {
        std::cerr << "Stream buffer too small" << std::endl;
        return 0;
    }

    const uint8_t* buf = (const uint8_t*) data;
    const uint8_t* header_ptr  = buf;
    const uint8_t* payload_ptr = buf + HEADER_SIZE;
    size_t payload_size = size_bytes - HEADER_SIZE - TRAILER_SIZE;
    const uint8_t* trailer_ptr = buf + size_bytes - TRAILER_SIZE;

    // Helper lambda to send a region directly (zero-copy for page-aligned buffers)
    auto sendRegion = [&](const uint8_t* src, size_t len) -> int {
        const size_t chunk_size = 32 * INSTR_BUF_SIZE;
        size_t remaining = len;
        while(remaining > 0) {
            size_t to_send = remaining < chunk_size ? remaining : chunk_size;
            int rc = iface->sendDataDirect(src, to_send);
            if(rc) return rc;
            src += to_send;
            remaining -= to_send;
        }
        return 0;
    };

    size_t total_payload;

    if(loop_count == 1) {
        // Single iteration: send the entire buffer as one contiguous transfer
        if(sendRegion(buf, size_bytes)) {
            std::cerr << "Stream data send failed" << std::endl;
            return 0;
        }
        total_payload = payload_size;
    } else {
        // Multiple iterations: split header / payload×N / trailer
        if(sendRegion(header_ptr, HEADER_SIZE)) {
            std::cerr << "Failed to send config header" << std::endl;
            return 0;
        }
        for(uint32_t i = 0; i < loop_count; i++) {
            if(payload_size > 0 && sendRegion(payload_ptr, payload_size)) {
                std::cerr << "Stream data send failed on iteration " << i << std::endl;
                return 0;
            }
        }
        if(sendRegion(trailer_ptr, TRAILER_SIZE)) {
            std::cerr << "Failed to send end marker" << std::endl;
            return 0;
        }
        total_payload = payload_size * loop_count;
    }
    std::cout << "Sent " << total_payload << " bytes of payload (" 
              << loop_count << " iteration(s))" << std::endl;

    // --- Step 3: Receive results via C2H ---
    // The readback engine may send multiple packets (e.g. HBM read data
    // from RD commands) before the final flush packet with the cycle count.
    // Loop until we see the flush bit, then extract the cycle count.

    uint64_t cycle_count = 0;
    bool done = false;
    while (!done) {
        int recvd = iface->recvData((void*)xdma_recv_buf, 32);
        if (recvd != 32) {
            std::cerr << "streamTest: failed to receive metadata" << std::endl;
            return 0;
        }
        uint64_t* meta = (uint64_t*)xdma_recv_buf;
        int fifo_entries = (int)(meta[0] & 0xFFF);
        bool is_flush = (meta[3] >> 63) & 1;
        int data_bytes = fifo_entries * 32;

        if (data_bytes > 0) {
            recvd = iface->recvData((void*)xdma_recv_buf, data_bytes);
            if (recvd != data_bytes) {
                std::cerr << "streamTest: failed to receive data payload" << std::endl;
                return 0;
            }

            // Look for cycle count: {8{stream_cycle_count}} across 2 FIFO entries
            if (is_flush && fifo_entries >= 2) {
                for (int e = 0; e <= fifo_entries - 2; e += 2) {
                    uint64_t* w = (uint64_t*)((uint8_t*)xdma_recv_buf + e * 32);
                    if (w[0] == 0) continue;
                    bool all_same = true;
                    for (int j = 1; j < 8; j++) {
                        if (w[j] != w[0]) { all_same = false; break; }
                    }
                    if (all_same) {
                        cycle_count = w[0];
                        break;
                    }
                }
            }
        }

        if (is_flush)
            done = true;
    }

    std::cout << "Stream test completed in " << cycle_count
              << " FPGA cycles (" << (double)cycle_count / 150e6 * 1e6
              << " us at 150 MHz)" << std::endl;

    return cycle_count;
}

// ---------------------------------------------------------------
// BRAM trace-replay control
//
// Sideband / beat protocol (must match bram_replay.v):
//   enter mode : 32B word, byte[9]=0x02  (sets h2c_tdata[`INSTR_WIDTH+9])
//   beat type  : byte[31] of each 256-bit beat
//                0=CONFIG 1=ENTRY 2=LOAD_DONE 3=STOP
//   CONFIG     : [0:3]=trace_len  [4:7]=num_iters  [20]=wr_en  [24:27]=wr_pattern
//   ENTRY      : [0:3]=command    [4:7]=timestamp   (routed to bank ts mod 4)
// byte[8] (h2c_tdata[`INSTR_WIDTH], the reset bit) is left 0 in every beat.
// ---------------------------------------------------------------

int SoftMCPlatform::loadTraceAndRun(const uint64_t* timestamps, const uint32_t* commands,
                                    size_t n_entries, uint64_t trace_length_slots,
                                    uint32_t num_iterations, uint32_t wr_pattern,
                                    bool wr_data_enable)
{
    if(is_dummy) {
        std::cerr << "loadTraceAndRun not available in dummy mode" << std::endl;
        return -1;
    }
    if(n_entries == 0) {
        std::cerr << "loadTraceAndRun: empty trace" << std::endl;
        return -1;
    }

    const size_t  BEAT        = 32;
    const uint8_t TYPE_CONFIG = 0, TYPE_ENTRY = 1, TYPE_DONE = 2;

    // --- Step 1: enter BRAM replay mode (sideband bit `INSTR_WIDTH+9 = byte 9 bit 1) ---
    memset(instr_buf, 0, 32);
    ((uint8_t*) instr_buf)[9] = (uint8_t) 0x02;
    int sent = iface->sendData(instr_buf, 32);
    memset(instr_buf, 0, 32);
    if(sent) {
        std::cerr << "Could not enter BRAM replay mode!" << std::endl;
        return -1;
    }
    std::cout << "Entered BRAM replay mode" << std::endl;

    // --- Step 2: build the load buffer [CONFIG][ENTRY x n][DONE] ---
    size_t n_beats   = 1 + n_entries + 1;
    size_t buf_bytes = n_beats * BEAT;
    void*  lbuf      = nullptr;
    if(posix_memalign(&lbuf, 4096, buf_bytes) != 0 || !lbuf) {
        std::cerr << "loadTraceAndRun: allocation failed" << std::endl;
        return -1;
    }
    memset(lbuf, 0, buf_bytes);
    uint8_t* p = (uint8_t*) lbuf;

    // CONFIG beat
    *(uint32_t*)(p + 0)  = (uint32_t) trace_length_slots;
    *(uint32_t*)(p + 4)  = num_iterations;
    p[20]                = wr_data_enable ? 1 : 0;
    *(uint32_t*)(p + 24) = wr_pattern;
    p[31]                = TYPE_CONFIG;
    p += BEAT;

    // ENTRY beats
    for(size_t i = 0; i < n_entries; i++) {
        *(uint32_t*)(p + 0) = commands[i];
        *(uint32_t*)(p + 4) = (uint32_t) timestamps[i];
        p[31]               = TYPE_ENTRY;
        p += BEAT;
    }

    // LOAD_DONE beat (arms + auto-starts the engine)
    p[31] = TYPE_DONE;

    // --- Step 3: stream the load buffer to the FPGA ---
    const size_t   chunk_size = 32 * INSTR_BUF_SIZE;
    const uint8_t* src        = (const uint8_t*) lbuf;
    size_t         remaining  = buf_bytes;
    while(remaining > 0) {
        size_t to_send = remaining < chunk_size ? remaining : chunk_size;
        if(iface->sendDataDirect(src, to_send)) {
            std::cerr << "loadTraceAndRun: load transfer failed" << std::endl;
            free(lbuf);
            return -1;
        }
        src       += to_send;
        remaining -= to_send;
    }
    free(lbuf);

    std::cout << "Loaded " << n_entries << " commands into BRAM (" << buf_bytes
              << " bytes), trace_len=" << trace_length_slots << " slots, iterations="
              << (num_iterations ? std::to_string(num_iterations) : std::string("infinite"))
              << std::endl;
    return 0;
}

void SoftMCPlatform::stopTraceLoop()
{
    if(is_dummy) return;
    uint8_t stop[32];
    memset(stop, 0, 32);
    stop[31] = 3;   // TYPE_STOP : h2c_tdata[255:248] == 3
    iface->sendData(stop, 32);
    std::cout << "Requested graceful stop of BRAM replay" << std::endl;
}

uint64_t SoftMCPlatform::waitTraceDone()
{
    if(is_dummy) return 0;

    // Same C2H readback as streamTest(): bram_replay flushes {8 x cycle_count}
    // through the readback engine on completion (stream_result_valid bypasses
    // garbage_reads, so it always gets through). For an infinite run this blocks
    // until stopTraceLoop() lets the engine finish its current iteration.
    uint64_t cycle_count = 0;
    bool done = false;
    while(!done) {
        int recvd = iface->recvData((void*)xdma_recv_buf, 32);
        if(recvd != 32) {
            std::cerr << "waitTraceDone: failed to receive metadata" << std::endl;
            return 0;
        }
        uint64_t* meta = (uint64_t*) xdma_recv_buf;
        int  fifo_entries = (int)(meta[0] & 0xFFF);
        bool is_flush     = (meta[3] >> 63) & 1;
        int  data_bytes   = fifo_entries * 32;

        if(data_bytes > 0) {
            recvd = iface->recvData((void*)xdma_recv_buf, data_bytes);
            if(recvd != data_bytes) {
                std::cerr << "waitTraceDone: failed to receive data payload" << std::endl;
                return 0;
            }
            if(is_flush && fifo_entries >= 2) {
                for(int e = 0; e <= fifo_entries - 2; e += 2) {
                    uint64_t* w = (uint64_t*)((uint8_t*)xdma_recv_buf + e * 32);
                    if(w[0] == 0) continue;
                    bool all_same = true;
                    for(int j = 1; j < 8; j++) if(w[j] != w[0]) { all_same = false; break; }
                    if(all_same) { cycle_count = w[0]; break; }
                }
            }
        }
        if(is_flush) done = true;
    }
    return cycle_count;
}

uint64_t SoftMCPlatform::waitTraceDoneCapture(std::vector<uint8_t>& read_data,
                                              bool strip_cycle_count)
{
    read_data.clear();
    if(is_dummy) return 0;

    // Same C2H protocol as waitTraceDone()/streamExecute(), but we KEEP the HBM
    // read-data entries (cycle-count {8 x count} block is identified and stripped).
    // Requires garbage reads OFF so rd_valid data reaches the readback FIFO.
    uint64_t cycle_count = 0;
    bool done = false;
    while(!done) {
        int recvd = iface->recvData((void*)xdma_recv_buf, 32);
        if(recvd != 32) {
            std::cerr << "waitTraceDoneCapture: failed to receive metadata" << std::endl;
            return 0;
        }
        uint64_t* meta = (uint64_t*) xdma_recv_buf;
        int  fifo_entries = (int)(meta[0] & 0xFFF);
        bool is_flush     = (meta[3] >> 63) & 1;
        int  data_bytes   = fifo_entries * 32;

        if(data_bytes > 0) {
            recvd = iface->recvData((void*)xdma_recv_buf, data_bytes);
            if(recvd != data_bytes) {
                std::cerr << "waitTraceDoneCapture: failed to receive data payload" << std::endl;
                return 0;
            }

            // Locate the cycle-count block: 2 consecutive 32-byte entries whose
            // eight 64-bit lanes are all equal (= {8 x cycle_count}).
            int cc_start = -1;
            if(strip_cycle_count && is_flush && fifo_entries >= 2) {
                for(int e = 0; e <= fifo_entries - 2; e += 2) {
                    uint64_t* w = (uint64_t*)((uint8_t*)xdma_recv_buf + e * 32);
                    if(w[0] == 0) continue;
                    bool all_same = true;
                    for(int j = 1; j < 8; j++) if(w[j] != w[0]) { all_same = false; break; }
                    if(all_same) { cycle_count = w[0]; cc_start = e; break; }
                }
            }

            auto append = [&](int from, int count) {
                if(count <= 0) return;
                size_t bytes = (size_t)count * 32;
                size_t old   = read_data.size();
                read_data.resize(old + bytes);
                memcpy(read_data.data() + old, (uint8_t*)xdma_recv_buf + (size_t)from * 32, bytes);
            };
            if(cc_start >= 0) {
                append(0, cc_start);
                append(cc_start + 2, fifo_entries - cc_start - 2);
            } else {
                append(0, fifo_entries);
            }
        }
        if(is_flush) done = true;
    }
    return cycle_count;
}

void* SoftMCPlatform::prepareStreamData(const void* payload, size_t payload_bytes, size_t* out_total_bytes, uint32_t consume_divider)
{
    // Layout (all offsets relative to buffer start):
    //   [32 bytes config header: divider(4B) + reserved(28B)]
    //   [payload padded to 32-byte boundary]
    //   [32 bytes trailer: end marker (0xFFFFFFFF) in first 4B, rest zero]
    //
    // This layout allows streamTest() to split the buffer cleanly:
    //   header (first 32B) | payload (middle) | trailer (last 32B)
    // and send the payload region multiple times for loop_count > 1.

    const size_t HEADER_SIZE = 32;
    const size_t TRAILER_SIZE = 32;
    // Pad payload up to 32-byte boundary so the trailer starts at a clean split
    size_t payload_padded = (payload_bytes + 31) & ~(size_t)31;
    size_t total = HEADER_SIZE + payload_padded + TRAILER_SIZE;

    void* buf = nullptr;
    if(posix_memalign(&buf, 4096, total) != 0) {
        std::cerr << "prepareStreamData: allocation failed" << std::endl;
        *out_total_bytes = 0;
        return nullptr;
    }
    memset(buf, 0, total);

    // Config header: first 4 bytes = consume_divider, rest zero
    uint32_t* hdr = (uint32_t*) buf;
    hdr[0] = (consume_divider == 0) ? 1 : consume_divider;

    // Copy payload after header
    memcpy((uint8_t*)buf + HEADER_SIZE, payload, payload_bytes);

    // Write end marker as the first 4 bytes of the trailer block
    uint32_t* marker_ptr = (uint32_t*)((uint8_t*)buf + HEADER_SIZE + payload_padded);
    *marker_ptr = 0xFFFFFFFF;

    *out_total_bytes = total;
    return buf;
}

// ---------------------------------------------------------------
// 32-bit command encoding
// ---------------------------------------------------------------

uint32_t SoftMCPlatform::encodeCommand(uint8_t cmd_type, uint16_t row, uint8_t col,
                                       uint8_t bg, uint8_t bank, uint8_t pc, uint8_t ch)
{
    uint32_t w = 0;
    w |= ((uint32_t)(cmd_type & 0xF)) << 28;
    w |= ((uint32_t)(row & 0x3FFF))   << 14;
    w |= ((uint32_t)(col & 0x1F))     << 9;
    w |= ((uint32_t)(bg  & 0x3))      << 7;
    w |= ((uint32_t)(bank & 0x3))     << 5;
    w |= ((uint32_t)(pc  & 0x1))      << 4;
    w |= ((uint32_t)(ch  & 0xF));
    return w;
}

uint32_t SoftMCPlatform::encodeNOP()
{
    return encodeCommand(CMD_NOP, 0, 0, 0, 0, 0, 0);
}

// ---------------------------------------------------------------
// Stream command preparation & execution
// ---------------------------------------------------------------

void* SoftMCPlatform::prepareStreamCommands(const std::vector<uint32_t>& commands, size_t* out_total_bytes)
{
    // Layout must match what the FPGA stream mode expects (same as prepareStreamData):
    //   [32 bytes config header: consume_divider(4B) + reserved(28B)]
    //   [payload padded to 32-byte boundary]
    //   [32 bytes trailer: end marker (0xFFFFFFFF) in first 4B, rest zero]
    const size_t HEADER_SIZE = 32;
    const size_t TRAILER_SIZE = 32;
    size_t payload_bytes = commands.size() * sizeof(uint32_t);
    size_t payload_padded = (payload_bytes + 31) & ~(size_t)31;
    size_t total = HEADER_SIZE + payload_padded + TRAILER_SIZE;

    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, total) != 0) {
        std::cerr << "prepareStreamCommands: allocation failed" << std::endl;
        *out_total_bytes = 0;
        return nullptr;
    }
    memset(buf, 0, total);

    // Config header: consume_divider = 1 (process one command per cycle)
    uint32_t* hdr = (uint32_t*)buf;
    hdr[0] = 1;

    // Copy command payload after header
    memcpy((uint8_t*)buf + HEADER_SIZE, commands.data(), payload_bytes);

    // Write end marker as first 4 bytes of the trailer block
    uint32_t* marker = (uint32_t*)((uint8_t*)buf + HEADER_SIZE + payload_padded);
    *marker = 0xFFFFFFFF;

    *out_total_bytes = total;
    return buf;
}

void* SoftMCPlatform::prepareStreamCommands(const StreamProgram& prog, size_t* out_total_bytes)
{
    const auto& commands = prog.commands();
    const size_t HEADER_SIZE  = 32;   // 256 bits
    const size_t TRAILER_SIZE = 32;   // 256 bits
    const size_t BEAT_SIZE    = 32;   // 256 bits per AXI beat

    const bool wr_data_en = prog.has_any_write_data();

    // -- Count payload beats -------------------------------------------------
    // Commands are grouped in 8 (= one 256-bit AXI beat).
    // After each group, one 256-bit data beat per WR/WRA in that group.
    size_t num_groups     = (commands.size() + 7) / 8;
    size_t num_data_beats = 0;
    if (wr_data_en) {
        for (size_t g = 0; g < num_groups; g++) {
            for (size_t j = 0; j < 8; j++) {
                size_t idx = g * 8 + j;
                if (idx >= commands.size()) break;
                uint8_t ct = (commands[idx] >> 28) & 0xF;
                if (ct == CMD_WR || ct == CMD_WRA)
                    num_data_beats++;
            }
        }
    }

    size_t payload_beats = num_groups + num_data_beats;
    size_t payload_bytes = payload_beats * BEAT_SIZE;
    size_t total = HEADER_SIZE + payload_bytes + TRAILER_SIZE;

    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, total) != 0) {
        std::cerr << "prepareStreamCommands: allocation failed" << std::endl;
        *out_total_bytes = 0;
        return nullptr;
    }
    memset(buf, 0, total);

    // -- Header (32 bytes) ---------------------------------------------------
    uint32_t* hdr = (uint32_t*)buf;
    hdr[0] = 1;                             // consume_divider (compat)
    hdr[1] = wr_data_en ? 1u : 0u;         // flags: bit 0 = wr_data_enable

    // -- Payload -------------------------------------------------------------
    uint8_t* dst = (uint8_t*)buf + HEADER_SIZE;
    const uint8_t* default_wd = prog.get_default_write_data();

    for (size_t g = 0; g < num_groups; g++) {
        // 8-command beat
        uint32_t group_cmds[8] = {};
        size_t   wr_indices[8];
        size_t   wr_count = 0;

        for (size_t j = 0; j < 8; j++) {
            size_t idx = g * 8 + j;
            if (idx < commands.size()) {
                group_cmds[j] = commands[idx];
                uint8_t ct = (commands[idx] >> 28) & 0xF;
                if (ct == CMD_WR || ct == CMD_WRA)
                    wr_indices[wr_count++] = idx;
            }
            // else 0 = NOP (already zero-initialized)
        }

        memcpy(dst, group_cmds, BEAT_SIZE);
        // Clear h2c_tdata[64] (byte[8] bit 0) — frontend.v interprets this
        // as a reset flag even during stream mode.
        dst[8] &= 0xFE;
        dst += BEAT_SIZE;

        // Data beats (one per write, in command order within the group)
        if (wr_data_en) {
            for (size_t w = 0; w < wr_count; w++) {
                const uint8_t* wd = prog.get_write_data(wr_indices[w]);
                if (wd)
                    memcpy(dst, wd, BEAT_SIZE);
                else if (default_wd)
                    memcpy(dst, default_wd, BEAT_SIZE);
                // else stays zero (memset above)

                // Clear bit 64 (reset flag) on data beats too
                dst[8] &= 0xFE;
                dst += BEAT_SIZE;
            }
        }
    }

    // -- Trailer (32 bytes: end marker in first 4 bytes) ---------------------
    uint32_t* marker = (uint32_t*)dst;
    *marker = 0xFFFFFFFF;

    *out_total_bytes = total;
    return buf;
}

int SoftMCPlatform::streamExecute(const void* data, size_t size_bytes,
                                  std::vector<uint8_t>& read_data, uint64_t& cycle_count)
{
    read_data.clear();
    cycle_count = 0;

    if (is_dummy) {
        std::cerr << "streamExecute not available in dummy mode" << std::endl;
        return -1;
    }

    // Ensure any receiver thread from a prior execute() has finished
    // to avoid concurrent reads on the C2H descriptor.
    if (receiver.joinable())
        receiver.join();

    // Step 1: Enter stream mode via sideband command (bit 72)
    //fprintf(stderr, "[DBG] streamExecute: entering stream mode...\n");
    memset(instr_buf, 0, 32);
    ((uint8_t*) instr_buf)[9] = (uint8_t)0x01;  // bit 72
    int sent = iface->sendData(instr_buf, 32);
    memset(instr_buf, 0, 32);
    if (sent) {
        std::cerr << "Could not enter stream mode!" << std::endl;
        return -1;
    }
    //fprintf(stderr, "[DBG] streamExecute: stream mode entered OK\n");

    // Step 2: Send the stream command buffer
    //fprintf(stderr, "[DBG] streamExecute: sending %zu bytes of H2C data...\n", size_bytes);

    // Hex dump full buffer (commented out)
    //{
    //    const uint8_t* d = (const uint8_t*)data;
    //    for (size_t off = 0; off < size_bytes; off += 32) {
    //        fprintf(stderr, "[DBG]   %04zx: ", off);
    //        for (size_t b = 0; b < 32 && off + b < size_bytes; b++)
    //            fprintf(stderr, "%02x", d[off + b]);
    //        fprintf(stderr, "  |");
    //        for (size_t w = 0; w < 8 && (off + w*4 + 3) < size_bytes; w++) {
    //            uint32_t val;
    //            memcpy(&val, d + off + w*4, 4);
    //            fprintf(stderr, " %08x", val);
    //        }
    //        fprintf(stderr, "\n");
    //    }
    //}

    const uint8_t* src = (const uint8_t*)data;
    size_t remaining = size_bytes;
    size_t chunk_num = 0;
    while (remaining > 0) {
        size_t to_send = remaining < 32 * INSTR_BUF_SIZE ? remaining : 32 * INSTR_BUF_SIZE;
        int rc = iface->sendDataDirect(src, to_send);
        if (rc) {
            std::cerr << "Stream data send failed" << std::endl;
            return -1;
        }
        //fprintf(stderr, "[DBG]   chunk %zu: sent %zu bytes, %zu remaining\n",
        //        chunk_num++, to_send, remaining - to_send);
        chunk_num++;
        src += to_send;
        remaining -= to_send;
    }
    //fprintf(stderr, "[DBG] streamExecute: all H2C data sent\n");

    // Step 3: Receive results via C2H
    // The readback engine will send metadata+data packets for HBM read data,
    // then a final flush packet with the cycle count (from stream_result_valid).
    // We keep reading packets until we get the flush packet (MSBit set in metadata).

    bool done = false;
    int pkt_num = 0;
    while (!done) {
        // Receive 32-byte metadata header
        //fprintf(stderr, "[DBG] streamExecute: waiting for C2H metadata packet %d...\n", pkt_num);
        int recvd = iface->recvData((void*)xdma_recv_buf, 32);
        if (recvd != 32) {
            std::cerr << "Failed to receive metadata (got " << recvd << " bytes)" << std::endl;
            return -1;
        }
        uint64_t* meta = (uint64_t*)xdma_recv_buf;
        int fifo_entries = (int)(meta[0] & 0xFFF);
        // Flush/last bit is at bit 255 of the 256-bit AXI word = bit 63 of meta[3]
        bool is_flush = (meta[3] >> 63) & 1;
        int data_bytes = fifo_entries * 32;  // each FIFO entry = 256 bits = 32 bytes
        //fprintf(stderr, "[DBG]   pkt %d: fifo_entries=%d, data_bytes=%d, is_flush=%d, meta={0x%016lx, 0x%016lx, 0x%016lx, 0x%016lx}\n",
        //        pkt_num, fifo_entries, data_bytes, (int)is_flush, meta[0], meta[1], meta[2], meta[3]);

        if (data_bytes > 0) {
            //fprintf(stderr, "[DBG]   pkt %d: reading %d bytes of data payload...\n", pkt_num, data_bytes);
            recvd = iface->recvData((void*)xdma_recv_buf, data_bytes);
            if (recvd != data_bytes) {
                std::cerr << "Failed to receive data payload (got " << recvd << " of " << data_bytes << " bytes)" << std::endl;
                return -1;
            }
            //fprintf(stderr, "[DBG]   pkt %d: data payload received OK\n", pkt_num);

            // Find and extract the cycle count from the data.
            // The readback engine injects {8{stream_cycle_count}} (512 bits)
            // which becomes 2 FIFO entries (64 bytes) of identical uint64 values.
            // Due to FIFO ordering, this may appear BEFORE or AFTER HBM read data
            // (depends on whether stream_done fires before rd_valid returns).
            // Scan all 2-entry aligned windows to find it.
            int data_entries = fifo_entries;
            int cc_start_entry = -1;
            if (is_flush && data_entries >= 2) {
                for (int e = 0; e <= data_entries - 2; e += 2) {
                    uint64_t* w = (uint64_t*)((uint8_t*)xdma_recv_buf + e * 32);
                    if (w[0] == 0) continue;
                    bool all_same = true;
                    for (int j = 1; j < 8; j++) {
                        if (w[j] != w[0]) { all_same = false; break; }
                    }
                    if (all_same) {
                        cycle_count = w[0];
                        cc_start_entry = e;
                        break;
                    }
                }
            }

            // Append HBM read data, skipping the cycle count block
            auto appendEntries = [&](int from, int count) {
                if (count <= 0) return;
                size_t bytes = count * 32;
                size_t old_sz = read_data.size();
                read_data.resize(old_sz + bytes);
                memcpy(read_data.data() + old_sz,
                       (uint8_t*)xdma_recv_buf + from * 32, bytes);
            };
            if (cc_start_entry >= 0) {
                appendEntries(0, cc_start_entry);
                appendEntries(cc_start_entry + 2, data_entries - cc_start_entry - 2);
            } else {
                appendEntries(0, data_entries);
            }
        }

        if (is_flush)
            done = true;
        pkt_num++;
    }
    //fprintf(stderr, "[DBG] streamExecute: C2H receive loop done after %d packets, cycle_count=%lu, read_data=%zu bytes\n",
    //        pkt_num, cycle_count, read_data.size());

    return 0;
}



void SoftMCPlatform::readRegisterDump()
{
  if(is_dummy)
    return;

  /** The author decided to use printfs explicitly within this function
   * because printing unsigned hex bytes is uglier with stdio
   */
  uint8_t readData[64];
  this -> receiveData((void*)readData, 64); // first read wdata content
  printf("WDATA: 0x");
  for(int i = 63 ; i >= 0 ; i--)
    printf("%x", readData[i]);
  printf("\n");
  this -> receiveData((void*)readData, 64); // read register content
  for(int r = 0 ; r < 16 ; r++)
  {
    printf("R%d: 0x", r);
    printf("%x", ((uint32_t*)readData)[r]);
    printf("\n");
  }
}

void SoftMCPlatform::setDimmSelect(int new_dimm_select) {
    dimm_select = new_dimm_select;
}

int SoftMCPlatform::getDimmSelect() const {
    return dimm_select;
}

/* Register access helper */
#define REG32(base, addr) (*(volatile uint32_t *)((uint8_t *)base + addr))

int SoftMCPlatform::initializeMonitoring()
{
    if(is_dummy) {
        std::cout << "Monitoring not available in dummy mode" << std::endl;
        return SOFTMC_SUCCESS;
    }

    if(monitoring_initialized) {
        std::cout << "Monitoring already initialized" << std::endl;
        return SOFTMC_SUCCESS;
    }

    off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);
    monitor_map_size = (HBM_1V2_I_INS_REG - map_phys_base) + sizeof(uint32_t);
    monitor_map_size = (monitor_map_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

    monitor_fd = open("/dev/xdma0_user", O_RDWR | O_SYNC);
    if (monitor_fd < 0) {
        std::cerr << "Failed to open /dev/xdma0_user for monitoring: " << strerror(errno) << std::endl;
        return SOFTMC_ERR;
    }

    monitor_map_base = mmap(NULL, monitor_map_size, PROT_READ | PROT_WRITE, MAP_SHARED, monitor_fd, map_phys_base);
    if (monitor_map_base == MAP_FAILED) {
        std::cerr << "Failed to mmap monitoring registers: " << strerror(errno) << std::endl;
        close(monitor_fd);
        monitor_fd = -1;
        return SOFTMC_ERR;
    }

    monitoring_initialized = true;

    // Store the physical base address for register access
    volatile uint32_t *reset_reg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (MB_RESETN_REG - map_phys_base));
    volatile uint32_t *ctrl_reg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (CONTROL_REG_ADDR - map_phys_base));
    volatile uint32_t *status_reg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HOST_STATUS2_REG_ADDR - map_phys_base));

    // Reset sequence
    *reset_reg = 0x0;
    usleep(10000);  // 10ms delay
    *reset_reg = 0x1;
    usleep(1000000); // 1000ms delay

    // Wait for ready
    int wait_count = 0;
    while (!(*status_reg & 0x1) && wait_count < 100) {
        usleep(10000);  // 10ms delay
        wait_count++;
    }

    if (wait_count >= 100) {
        std::cerr << "Timeout waiting for monitoring system to be ready" << std::endl;
        return SOFTMC_ERR;
    }

    // Enable HBM monitoring (set bit 27 in control register)
    *ctrl_reg |= (1 << 27);

    std::cout << "HBM monitoring initialized successfully" << std::endl;
    return SOFTMC_SUCCESS;
}

void SoftMCPlatform::resetMonitoring()
{
    if(is_dummy || !monitoring_initialized) {
        return;
    }

    off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);
    volatile uint32_t *ctrl_reg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (CONTROL_REG_ADDR - map_phys_base));

    // Reset sequence
    *ctrl_reg |= 0x1;
    usleep(10000);  // 10ms delay

    std::cout << "HBM monitoring reset" << std::endl;
}

void SoftMCPlatform::setCSVFilePath(const std::string& path)
{
    // Delete the file if it already exists
    if (std::remove(path.c_str()) == 0 || errno == ENOENT) {
        // File was deleted successfully or didn't exist
    }
    
    csv_file_path = path;
    csv_header_written = false;  // Reset header flag for new file
}

std::string SoftMCPlatform::getCSVFilePath() const
{
    return csv_file_path;
}

uint32_t SoftMCPlatform::getAverageCurrent()
{
    if(is_dummy) {
        return 0;
    }

    if(!monitoring_initialized) {
        return 0;
    }

    off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);
    volatile uint32_t *current_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_I_AVG_REG - map_phys_base));
    return *current_avg;
}

void SoftMCPlatform::readAndPrintHBMMetrics()
{
    if(is_dummy) {
        std::cout << "Monitoring not available in dummy mode" << std::endl;
        return;
    }

    if(!monitoring_initialized) {
        std::cerr << "Monitoring not initialized" << std::endl;
        return;
    }

    off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);
    
    // Read temp1 registers
    volatile uint32_t *temp1_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP1_INS_REG - map_phys_base));
    volatile uint32_t *temp1_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP1_MAX_REG - map_phys_base));
    volatile uint32_t *temp1_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP1_AVG_REG - map_phys_base));

    // Read temp2 registers
    volatile uint32_t *temp2_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP2_INS_REG - map_phys_base));
    volatile uint32_t *temp2_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP2_MAX_REG - map_phys_base));
    volatile uint32_t *temp2_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP2_AVG_REG - map_phys_base));

    // Read voltage registers
    volatile uint32_t *voltage_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_INS_REG - map_phys_base));
    volatile uint32_t *voltage_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_MAX_REG - map_phys_base));
    volatile uint32_t *voltage_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_AVG_REG - map_phys_base));

    // Read current registers
    volatile uint32_t *current_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_I_INS_REG - map_phys_base));
    volatile uint32_t *current_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_I_MAX_REG - map_phys_base));
    volatile uint32_t *current_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_1V2_I_AVG_REG - map_phys_base));

    volatile uint32_t *pex_12v_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_12V_INS_REG - map_phys_base));
    volatile uint32_t *pex_12v_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_12V_MAX_REG - map_phys_base));
    volatile uint32_t *pex_12v_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_12V_AVG_REG - map_phys_base));

    volatile uint32_t *pex_12v_i_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_12VPEX_IN_INS_REG - map_phys_base));
    volatile uint32_t *pex_12v_i_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_12VPEX_IN_MAX_REG - map_phys_base));
    volatile uint32_t *pex_12v_i_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_12VPEX_IN_AVG_REG - map_phys_base));

    volatile uint32_t *pex_3v3_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_3V3_INS_REG - map_phys_base));
    volatile uint32_t *pex_3v3_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_3V3_MAX_REG - map_phys_base));
    volatile uint32_t *pex_3v3_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (PEX_3V3_AVG_REG - map_phys_base));

    volatile uint32_t *pex_3v3_i_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_3V3PEX_IN_INS_REG - map_phys_base));
    volatile uint32_t *pex_3v3_i_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_3V3PEX_IN_MAX_REG - map_phys_base));
    volatile uint32_t *pex_3v3_i_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (I_3V3PEX_IN_AVG_REG - map_phys_base));

    volatile uint32_t *vccint_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_INS_REG - map_phys_base));
    volatile uint32_t *vccint_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_MAX_REG - map_phys_base));
    volatile uint32_t *vccint_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_AVG_REG - map_phys_base));

    volatile uint32_t *vccint_i_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_I_INS_REG - map_phys_base));
    volatile uint32_t *vccint_i_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_I_MAX_REG - map_phys_base));
    volatile uint32_t *vccint_i_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_I_AVG_REG - map_phys_base));

    volatile uint32_t *vccint_io_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_INS_REG - map_phys_base));
    volatile uint32_t *vccint_io_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_MAX_REG - map_phys_base));
    volatile uint32_t *vccint_io_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_AVG_REG - map_phys_base));

    volatile uint32_t *vccint_io_i_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_I_INS_REG - map_phys_base));
    volatile uint32_t *vccint_io_i_max = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_I_MAX_REG - map_phys_base));
    volatile uint32_t *vccint_io_i_avg = (volatile uint32_t *)((uint8_t *)monitor_map_base + (VCCINT_IO_I_AVG_REG - map_phys_base));

    volatile uint32_t power_ins = (*pex_12v_ins) * (*pex_12v_i_ins) + (*pex_3v3_ins) * (*pex_3v3_i_ins) - (*vccint_ins) * (*vccint_i_ins) - (*vccint_io_ins) * (*vccint_io_i_ins) - (*voltage_ins) * (*current_ins);
    volatile uint32_t power_max = (*pex_12v_max) * (*pex_12v_i_max) + (*pex_3v3_max) * (*pex_3v3_i_max) - (*vccint_max) * (*vccint_i_max) - (*vccint_io_max) * (*vccint_io_i_max) - (*voltage_max) * (*current_max);
    volatile uint32_t power_avg = (*pex_12v_avg) * (*pex_12v_i_avg) + (*pex_3v3_avg) * (*pex_3v3_i_avg) - (*vccint_avg) * (*vccint_i_avg) - (*vccint_io_avg) * (*vccint_io_i_avg) - (*voltage_avg) * (*current_avg);
    power_ins = (double) power_ins / 1000.0; // convert to mW
    power_max = (double) power_max / 1000.0; // convert to mW
    power_avg = (double) power_avg / 1000.0; // convert to mW

    // average power was experimentally measured to be around 8170mW, so subtracting that
    // power_ins = std::max((int) power_ins - 8170, 0);
    // power_max = std::max((int) power_max - 8170, 0);
    // power_avg = std::max((int) power_avg - 8170, 0);

    // Print all metrics to console
    // printf("===== HBM Monitoring Metrics =====\n");
    // printf("HBM 1.2V Voltage (mV):\n");
    // printf("  Instantaneous:         %u\n", *voltage_ins);
    // printf("  Maximum:               %u\n", *voltage_max);
    // printf("  Average:               %u\n", *voltage_avg);
    // printf("HBM 1.2V Current (mA):\n");
    // printf("  Instantaneous:         %u\n", *current_ins);
    // printf("  Maximum:               %u\n", *current_max);
    // printf("  Average:               %u\n", *current_avg);
    // printf("===================================\n");

    // Append metrics to CSV file
    std::ofstream csv_file(csv_file_path, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_file_path << std::endl;
        return;
    }

    // Write header if this is the first write
    if (!csv_header_written) {
        csv_file << "Timestamp,Temp1_Ins(Temp),Temp1_Max(Temp),Temp1_Avg(Temp),Temp2_Ins(Temp),Temp2_Max(Temp),Temp2_Avg(Temp),Voltage_Ins(mV),Voltage_Max(mV),Voltage_Avg(mV),Current_Ins(mA),Current_Max(mA),Current_Avg(mA),Power_VDD_Ins(mW),Power_VDD_Max(mW),Power_VDD_Avg(mW),Power_VPP_Ins(mW),Power_VPP_Max(mW),Power_VPP_Avg(mW)\n";
        csv_header_written = true;
    }

    // Get current timestamp
    struct timeval tv;
    gettimeofday(&tv, NULL);

    // Write data row
    csv_file << tv.tv_sec << "." << tv.tv_usec << ","
             << *temp1_ins << "," << *temp1_max << "," << *temp1_avg << ","
             << *temp2_ins << "," << *temp2_max << "," << *temp2_avg << ","
             << *voltage_ins << "," << *voltage_max << "," << *voltage_avg << ","
             << *current_ins << "," << *current_max << "," << *current_avg << ","
            //  << (*pex_12v_ins) * (*pex_12v_i_ins) / 1000.0 << "," << (*pex_12v_max) * (*pex_12v_i_max) / 1000.0 << "," << (*pex_12v_avg) * (*pex_12v_i_avg) / 1000.0 << ","
            //  << (*pex_3v3_ins) * (*pex_3v3_i_ins) / 1000.0 << "," << (*pex_3v3_max) * (*pex_3v3_i_max) / 1000.0 << "," << (*pex_3v3_avg) * (*pex_3v3_i_avg) / 1000.0 << ","
            //  << (*vccint_ins) * (*vccint_i_ins) << "," << (*vccint_max) * (*vccint_i_max) << "," << (*vccint_avg) * (*vccint_i_avg) << ","
            //  << (*vccint_io_ins) * (*vccint_io_i_ins) << "," << (*vccint_io_max) * (*vccint_io_i_max) << "," << (*vccint_io_avg) * (*vccint_io_i_avg) << ","
             << (*voltage_ins) * (*current_ins) / 1000.0 << "," << (*voltage_max) * (*current_max) / 1000.0 << "," << (*voltage_avg) * (*current_avg) / 1000.0 << ","
             << power_ins << "," << power_max << "," << power_avg << "\n";
    
    csv_file.close();
    std::cout << "HBM metrics appended to: " << csv_file_path << std::endl;
}

void SoftMCPlatform::startMetricsThread(uint64_t sleep_ms)
{
    if (metrics_thread_running) {
        std::cerr << "Metrics thread is already running" << std::endl;
        return;
    }

    metrics_thread_running = true;

    metrics_thread = std::thread([this, sleep_ms]() {
        // Helper lambda: wait until the sliding window of `window` readings
        // has a max-min range <= `range_ma`.  Returns false if stopped externally.
        auto wait_for_stable = [&](int window, uint32_t range_ma,
                                   const std::string& phase_tag) -> bool {
            std::vector<uint32_t> readings;

            std::cout << "[" << phase_tag << "] Waiting for power stabilization ("
                      << range_ma << " mA range for "
                      << window   << " consecutive cycles)..."
                      << std::endl;

            while (metrics_thread_running) {
                uint32_t current_avg = getAverageCurrent();
                readings.push_back(current_avg);

                if ((int)readings.size() > window)
                    readings.erase(readings.begin());

                if ((int)readings.size() == window) {
                    uint32_t min_val = *std::min_element(readings.begin(),
                                                         readings.end());
                    uint32_t max_val = *std::max_element(readings.begin(),
                                                         readings.end());

                    std::cout << "[" << phase_tag << "] Stabilization check: range = "
                              << (max_val - min_val) << " mA  (current: "
                              << current_avg << " mA, window min: " << min_val
                              << ", window max: " << max_val << ")" << std::endl;

                    if (max_val - min_val <= range_ma) {
                        std::cout << "[" << phase_tag << "] Power stabilized! Range "
                                  << (max_val - min_val) << " mA <= "
                                  << range_ma << " mA threshold."
                                  << std::endl;
                        return true;
                    }
                } else {
                    std::cout << "[" << phase_tag << "] Collecting readings ("
                              << readings.size() << "/"
                              << window << "), current: "
                              << current_avg << " mA" << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
            return false;  // Stopped externally
        };

        // =============================================
        // Phase 1a: Coarse stabilization
        //   Smaller window, larger mA range.
        // =============================================
        if (!wait_for_stable(COARSE_STABILIZATION_WINDOW,
                             COARSE_STABILIZATION_RANGE_MA,
                             "Phase 1a – coarse"))
            return;

        // =============================================
        // Phase 1b: Fine stabilization
        //   Longer window, narrower mA range.
        // =============================================
        if (!wait_for_stable(FINE_STABILIZATION_WINDOW,
                             FINE_STABILIZATION_RANGE_MA,
                             "Phase 1b – fine"))
            return;

        // =============================================
        // Phase 2: Reset metrics and collect for
        //          COLLECTION_DURATION_S seconds.
        //          Only this data goes into the CSV.
        // =============================================
        std::cout << "[Phase 2] Resetting monitoring and collecting data for "
                  << COLLECTION_DURATION_S << " seconds..." << std::endl;

        resetMonitoring();

        // Clear the CSV file so only post-stabilization data is recorded
        setCSVFilePath(csv_file_path);

        auto collection_start = std::chrono::steady_clock::now();
        int measurement_count = 0;
        while (metrics_thread_running) {
            readAndPrintHBMMetrics();
            measurement_count++;

            // if (measurement_count % METRICS_RESET_INTERVAL == 0) {
            //     std::cout << "[Phase 2] Resetting monitoring after "
            //               << measurement_count << " measurements" << std::endl;
            //     resetMonitoring();
            // }

            auto elapsed = std::chrono::steady_clock::now() - collection_start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= COLLECTION_DURATION_S) {
                std::cout << "[Phase 2] Collection complete ("
                          << COLLECTION_DURATION_S << " seconds)."
                          << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        if (!metrics_thread_running) return;  // Stopped externally

        // =============================================
        // Phase 3: Stop the FPGA program and reset.
        // =============================================
        std::cout << "[Phase 3] Stopping program and resetting FPGA..."
                  << std::endl;

        // Signal the receiver thread to stop
        receiver_should_stop = true;

        // Disable garbage reads and close descriptor to unblock receiver
        if (garbage_reads_enabled && iface != nullptr) {
            iface->disableGarbageReads();
            iface->closeReceiveDescriptor();
        }

        // Wait briefly for the receiver thread to notice the stop signal
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Reset the FPGA
        reset_fpga();

        metrics_thread_running = false;
        std::cout << "[Phase 3] FPGA reset complete. Measurement done."
                  << std::endl;
    });

    std::cout << "Metrics thread started" << std::endl;
}

void SoftMCPlatform::startMetricsThreadFixedDuration(uint64_t sleep_ms, int duration_s)
{
    if (metrics_thread_running) {
        std::cerr << "Metrics thread is already running" << std::endl;
        return;
    }

    metrics_thread_running = true;

    metrics_thread = std::thread([this, sleep_ms, duration_s]() {
        std::cout << "[Fixed-duration] Recording metrics for "
                  << duration_s << " seconds..." << std::endl;

        auto start = std::chrono::steady_clock::now();
        int measurement_count = 0;
        while (metrics_thread_running) {
            readAndPrintHBMMetrics();
            measurement_count++;

            if (measurement_count % METRICS_RESET_INTERVAL == 0) {
                std::cout << "[Fixed-duration] Resetting monitoring after "
                          << measurement_count << " measurements" << std::endl;
                resetMonitoring();
            }

            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= duration_s) {
                std::cout << "[Fixed-duration] Collection complete ("
                          << duration_s << " seconds)." << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        if (!metrics_thread_running) return;  // Stopped externally

        // Stop the FPGA program and reset
        std::cout << "[Fixed-duration] Stopping program and resetting FPGA..."
                  << std::endl;

        receiver_should_stop = true;

        if (garbage_reads_enabled && iface != nullptr) {
            iface->disableGarbageReads();
            iface->closeReceiveDescriptor();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        reset_fpga();

        metrics_thread_running = false;
        std::cout << "[Fixed-duration] FPGA reset complete. Measurement done."
                  << std::endl;
    });

    std::cout << "Metrics thread started (fixed-duration mode, "
              << duration_s << "s)" << std::endl;
}

void SoftMCPlatform::stopMetricsThread()
{
    if (!metrics_thread_running) {
        std::cerr << "Metrics thread is not running" << std::endl;
        return;
    }

    metrics_thread_running = false;
    if (metrics_thread.joinable()) {
        metrics_thread.join();
    }

    std::cout << "Metrics thread stopped" << std::endl;
}

/* ===================== Temperature Watchdog ===================== */

void SoftMCPlatform::setMaxTemperature(int temp)
{
    max_temperature = temp;
    std::cout << "Max temperature threshold set to " << max_temperature << " °C" << std::endl;
}

int SoftMCPlatform::getMaxTemperature() const
{
    return max_temperature;
}

void SoftMCPlatform::startTemperatureWatchdog(uint64_t check_interval_ms)
{
    if (is_dummy) {
        std::cout << "Temperature watchdog not available in dummy mode" << std::endl;
        return;
    }

    if (temp_watchdog_running) {
        std::cerr << "Temperature watchdog is already running" << std::endl;
        return;
    }

    if (!monitoring_initialized) {
        std::cerr << "Cannot start temperature watchdog: monitoring not initialized. "
                  << "Call initializeMonitoring() first." << std::endl;
        return;
    }

    over_temp_shutdown = false;
    temp_watchdog_running = true;

    temp_watchdog_thread = std::thread([this, check_interval_ms]() {
        std::cout << "[TempWatchdog] Started — threshold: " << max_temperature
                  << " °C, interval: " << check_interval_ms << " ms" << std::endl;

        off_t map_phys_base = MB_RESETN_REG & ~(PAGE_SIZE - 1);

        while (temp_watchdog_running) {
            volatile uint32_t *temp1_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP1_INS_REG - map_phys_base));
            volatile uint32_t *temp2_ins = (volatile uint32_t *)((uint8_t *)monitor_map_base + (HBM_TEMP2_INS_REG - map_phys_base));

            uint32_t t1 = *temp1_ins;
            uint32_t t2 = *temp2_ins;
            uint32_t t_max = std::max(t1, t2);

            if ((int)t_max >= max_temperature) {
                std::cerr << "\n[TempWatchdog] *** OVER-TEMPERATURE DETECTED ***" << std::endl;
                std::cerr << "[TempWatchdog] Temp1: " << t1 << " °C, Temp2: " << t2
                          << " °C  (threshold: " << max_temperature << " °C)" << std::endl;
                std::cerr << "[TempWatchdog] Resetting FPGA and stopping execution..." << std::endl;

                over_temp_shutdown = true;

                // Signal the receiver thread to stop
                receiver_should_stop = true;

                // Disable garbage reads and close descriptor to unblock receiver
                if (garbage_reads_enabled && iface != nullptr) {
                    iface->disableGarbageReads();
                    iface->closeReceiveDescriptor();
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(200));

                // Reset the FPGA
                reset_fpga();

                // Also stop the metrics thread if running
                if (metrics_thread_running) {
                    metrics_thread_running = false;
                }

                std::cerr << "[TempWatchdog] FPGA reset complete. Execution halted due to over-temperature."
                          << std::endl;

                temp_watchdog_running = false;
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms));
        }

        std::cout << "[TempWatchdog] Stopped" << std::endl;
    });

    std::cout << "Temperature watchdog started" << std::endl;
}

void SoftMCPlatform::stopTemperatureWatchdog()
{
    if (!temp_watchdog_running) {
        std::cerr << "Temperature watchdog is not running" << std::endl;
        return;
    }

    temp_watchdog_running = false;
    if (temp_watchdog_thread.joinable()) {
        temp_watchdog_thread.join();
    }

    std::cout << "Temperature watchdog stopped" << std::endl;
}