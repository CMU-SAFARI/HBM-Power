#pragma once

#include <list>
#include <memory>

#include "base/base.h"
#include "base/config.h"
#include "common/type.h"
#include "dram/dram_request.h"
#include "dram/dram_type.h"
#include "dram/pim_request.h"
#include "dram/pimkernel/pim_kernel.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"
#include "module/status.h"
#include "power.h"

namespace llm_system {

using pim_kernel_ptr = PIM_KERNEL::pim_kernel_ptr;

class DRAMInterface {
 public:
  friend class DataObject;

  using Ptr = std::shared_ptr<DRAMInterface>;

  [[nodiscard]] DRAMInterface::Ptr static Create(
      std::string config_path, const double memory_scale_factor) {
    return DRAMInterface::Ptr(
        new DRAMInterface(config_path, memory_scale_factor));
  }

  DRAMInterface(DRAMInterface &&) = default;
  DRAMInterface &operator=(DRAMInterface &&) = default;

  void HandleRequest(const std::list<DRAMRequest::Ptr> &requests,
                     cycle_t start_time_cycle);

  void run();
  ExecStatus &getExecStatus() { return exec_status; }
  PIMRequest &GeneratePIMCommand(const DRAMRequest::Ptr request,
                                 PIMRequest &pimrequest) const;
  void resetCounter();

  PIMHWConfig pim_hw_config;

  void setPIMHWConfig(ProcessorType type, int bandwidth_x) {
    pim_hw_config.type = type;
    pim_hw_config.bandwidth_x = bandwidth_x;
  }

  std::map<DRAMRequestType, std::string> dramreq_to_string = {};
  std::map<PIMCommandType, std::string> pimcmd_to_string = {};
  std::map<PIMOperandType, std::string> pimoperand_to_string = {};

  time_ns time;

 private:
  Ramulator::IFrontEnd *frontend;
  Ramulator::IMemorySystem *memory_system;

  ExecStatus exec_status;

  long frontend_tick;
  long mem_tick;
  long tick_mult;
  double memory_scale_factor;

  std::vector<std::function<pim_kernel_ptr>> kernel;

  DRAMInterface(std::string config_path,
                double memory_scale_factor);

  void updateStatus(const PIMRequest &pimrequest);
  void SendRequest(PIMRequest &pimrequest);

  void StringMapInit();

  std::vector<DRAMRequest> dram_request_log_;
  std::vector<DRAMRequest> pim_request_log_;
};

}  // namespace llm_system