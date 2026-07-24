#include <filesystem>
#include <fstream>
#include <iostream>

#include "base/exception.h"
#include "dram/memory_object.h"
#include "frontend/frontend.h"
// #include "rf_request.h"

namespace Ramulator {

namespace fs = std::filesystem;

void set_end_time(Request& req) {
  req.pimcmd.request->end = req.depart;
  for (int i = 0; i < (int)llm_system::DRAMCommandType::kMAX; i++) {
    req.pimcmd.request->issued_dram_cmd[i] += req.issued_dram_cmd[i];
  }
}

class PIMController : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd, PIMController, "PIMController",
      "PIM controller which generates PIM commands")

 private:
  struct Trace {
    bool is_write;
    Addr_t addr;
  };
  std::vector<Trace> m_trace;

  size_t m_trace_length = 0;
  size_t m_curr_trace_idx = 0;

  size_t m_trace_count = 0;
  Logger_t m_logger;

  int get_type(llm_system::PIMCommandType type) {
    switch (type) {
      case llm_system::PIMCommandType::kAdd:
      case llm_system::PIMCommandType::kSub:
      case llm_system::PIMCommandType::kMult:
      case llm_system::PIMCommandType::kMAC:
      case llm_system::PIMCommandType::kDRAM2RF:
        return (int)Request::Type::AllRead;
      case llm_system::PIMCommandType::kRF2DRAM:
        return (int)Request::Type::AllWrite;
      case llm_system::PIMCommandType::kRead:
        return (int)Request::Type::Read;
      case llm_system::PIMCommandType::kWrite:
        return (int)Request::Type::Write;
      default:
        assert(0);
    }
  }

 public:
  void init() override {
    // std::string trace_path_str = param<std::string>("path").desc("Path to the
    // load store trace file.").required();
    m_clock_ratio = param<uint>("clock_ratio").required();

    // m_logger = Logging::create_logger("PIM_Controller");
    //  m_logger->info("Loading trace file {} ...", trace_path_str);
    //  init_trace(trace_path_str);
    //  m_logger->info("Loaded {} lines.", m_trace.size());
  };

  // void send(llm_system::DRAMRequest &request) override{
  //   return;
  // }

  void tick() override {
    if (m_curr_trace_idx >= m_trace_length) {
      return;
    }
    const Trace& t = m_trace[m_curr_trace_idx];
    bool request_sent =
        m_memory_system->send({t.addr, t.is_write ? Request::Type::AllWrite
                                                  : Request::Type::AllRead});
    if (request_sent) {
      // m_curr_trace_idx = (m_curr_trace_idx + 1) % m_trace_length;
      m_curr_trace_idx++;
      m_trace_count++;
    }
  };

  void send(llm_system::PIMRequest& pimrequest) override {
    pimrequest.start = m_memory_system->m_clk;
    pimrequest.end = m_memory_system->m_clk;
    for (auto& cmd : pimrequest.command_queue) {
      auto type = get_type(cmd.pimcmd_type);
      m_memory_system->send(
          {cmd.addr_vec, get_type(cmd.pimcmd_type), cmd, &set_end_time});
    }
  }

  bool is_finished() override { return m_trace_count >= m_trace_length; };
};

}  // namespace Ramulator