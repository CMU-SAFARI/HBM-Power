#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus ActivationExecutionGPU(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  hw_metric total_flops = 0;
  hw_metric total_memory_size = 0;

  total_memory_size =
      gate_output->getSize() + input->getSize() + output->getSize();

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
                                  ProcessorType::GPU, DRAMRequestType::kRead,
                                  PIMOperandType::kDRAM, gate_output);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += 
        getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, gate_output);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus ActivationExecutionLogic(Device_Ptr device, Tensor_Ptr gate_output,
                                    Tensor_Ptr input, Tensor_Ptr output,
                                    bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  hw_metric total_flops = 0;
  hw_metric total_memory_size = 0;

  total_memory_size =
      gate_output->getSize() + input->getSize() + output->getSize();

  total_flops += total_memory_size;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
                                  ProcessorType::LOGIC, DRAMRequestType::kGEMV,
                                  PIMOperandType::kSrc, gate_output);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += 
        getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, gate_output);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus ActivationExecutionPIM(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  hw_metric total_flops = 0;
  hw_metric total_memory_size = 0;

  total_memory_size =
      gate_output->getSize() + input->getSize() + output->getSize();

  total_flops += total_memory_size;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
                                  ProcessorType::PIM, DRAMRequestType::kRead,
                                  PIMOperandType::kDRAM, gate_output);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::PIM,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::ACTIVATION, ProcessorType::PIM,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += 
        getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, gate_output);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);

    exec_status +=
        getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

}  // namespace llm_system