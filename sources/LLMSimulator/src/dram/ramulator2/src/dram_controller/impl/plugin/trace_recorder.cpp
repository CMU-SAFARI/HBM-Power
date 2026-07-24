#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <limits>
#include <unordered_map>
#include <vector>

#include "base/base.h"
#include "dram/pim_request.h"
#include "dram_controller/controller.h"
#include "dram_controller/plugin.h"

namespace Ramulator {

class TraceRecorder : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, TraceRecorder, "TraceRecorder", "CounterBasedTRR.")
  private:
    IDRAM* m_dram;

    std::filesystem::path m_trace_path; 
    Logger_t m_tracer;

    Clk_t m_clk = 0;

    std::map<llm_system::DRAMRequestType, std::string> dramreq_to_string = {};
    std::map<llm_system::PIMCommandType, std::string> pimcmd_to_string = {};
    std::map<llm_system::PIMOperandType, std::string> pimoperand_to_string = {};

   public:
    void init() override {
      m_trace_path =
          param<std::string>("path").desc("Path to the trace file").required();
      auto parent_path = m_trace_path.parent_path();
      std::filesystem::create_directories(parent_path);
      if (!(std::filesystem::exists(parent_path) &&
            std::filesystem::is_directory(parent_path))) {
        throw ConfigurationError("Invalid path to trace file: {}",
                                 parent_path.string());
      }

      dramreq_to_string[llm_system::DRAMRequestType::kRead] = "Read";
      dramreq_to_string[llm_system::DRAMRequestType::kWrite] = "Write";
      dramreq_to_string[llm_system::DRAMRequestType::kMove] = "Move";
      dramreq_to_string[llm_system::DRAMRequestType::kMult] = "Mult";
      dramreq_to_string[llm_system::DRAMRequestType::kAdd] = "Add";
      dramreq_to_string[llm_system::DRAMRequestType::kMAD] = "MAD";
      dramreq_to_string[llm_system::DRAMRequestType::kPMult] = "PMult";
      dramreq_to_string[llm_system::DRAMRequestType::kCMult] = "CMult";
      dramreq_to_string[llm_system::DRAMRequestType::kCAdd] = "CAdd";
      dramreq_to_string[llm_system::DRAMRequestType::kCMAD] = "CMAD";
      dramreq_to_string[llm_system::DRAMRequestType::kTensor] = "Tensor";
      dramreq_to_string[llm_system::DRAMRequestType::kTensor_Square] =
          "Tensor_Square";
      dramreq_to_string[llm_system::DRAMRequestType::kModup_Evkmult] =
          "Modup_Evkmult";
      dramreq_to_string[llm_system::DRAMRequestType::kModDownEpilogue] =
          "ModDownEpilogue";
      dramreq_to_string[llm_system::DRAMRequestType::kPMult_Accum] = "PMult_Accum";
      dramreq_to_string[llm_system::DRAMRequestType::kCMult_Accum] = "CMult_Accum";

      pimcmd_to_string[llm_system::PIMCommandType::kAdd] = "Add";
      pimcmd_to_string[llm_system::PIMCommandType::kSub] = "Sub";
      pimcmd_to_string[llm_system::PIMCommandType::kMult] = "Mult";
      pimcmd_to_string[llm_system::PIMCommandType::kMAC] = "MAC";
      pimcmd_to_string[llm_system::PIMCommandType::kDRAM2RF] = "DRAM2RF";
      pimcmd_to_string[llm_system::PIMCommandType::kRF2DRAM] = "RF2DRAM";
      pimcmd_to_string[llm_system::PIMCommandType::kRead] = "Read";
      pimcmd_to_string[llm_system::PIMCommandType::kWrite] = "Write";

      pimoperand_to_string[llm_system::PIMOperandType::kRF] = "RF";
      pimoperand_to_string[llm_system::PIMOperandType::kDRAM] = "DRAM";
      pimoperand_to_string[llm_system::PIMOperandType::kSrc] = "Src";
      pimoperand_to_string[llm_system::PIMOperandType::kPrecomputed] =
          "Precomputed";
      pimoperand_to_string[llm_system::PIMOperandType::kDest] = "Dest";
      pimoperand_to_string[llm_system::PIMOperandType::kModUp] = "ModUp";
      pimoperand_to_string[llm_system::PIMOperandType::kEvk] = "Evk";
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_ctrl = cast_parent<IDRAMController>();
      m_dram = m_ctrl->m_dram;

      auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("{}.ch{}", m_trace_path.string(), m_ctrl->m_channel_id), true);
      m_tracer = std::make_shared<spdlog::logger>(fmt::format("trace_recorder_ch{}", m_ctrl->m_channel_id), sink);
      m_tracer->set_pattern("%v");
      m_tracer->set_level(spdlog::level::trace);      
    };

    void update(bool request_found, ReqBuffer::iterator& req_it) override {
      m_clk++;

      if (request_found) {
        m_tracer->trace("{}, {}, {}, {}, {}, {}, {}", m_clk,
                        m_dram->m_commands(req_it->command),
                        fmt::join(req_it->addr_vec, ", "), req_it->addr,
                        pimcmd_to_string[req_it->pimcmd.pimcmd_type],
                        pimoperand_to_string[req_it->pimcmd.op_type],
                        dramreq_to_string[req_it->pimcmd.dramreq_type]);
      }

    };

};

}       // namespace Ramulator
