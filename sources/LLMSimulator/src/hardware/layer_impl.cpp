#include "hardware/layer_impl.h"

#include "dram/dram_interface.h"
#include "hardware/device.h"
#include "module/tensor.h"

namespace llm_system {

ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor) {
  CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type,
                                 tensor->getSize());
  ExecStatus exec_status;
  if (!device->checkExecutionCache(exec_status, key)) {
    DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
    dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
    device->run_ramulator(dram_request);
    exec_status = device->dram_interface->getExecStatus(); 
    device->addExecutionCache(exec_status, key);
  }

  return exec_status;
};

ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor) {
  
  ExecStatus exec_status;
  long total_size = tensor->getSize();
  device->run_ideal(dram_request_type, tensor);
  exec_status = device->dram_interface->getExecStatus(); 
  return exec_status;
};

}  // namespace llm_system