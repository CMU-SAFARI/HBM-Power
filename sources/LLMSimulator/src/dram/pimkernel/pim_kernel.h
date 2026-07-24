#pragma once

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "base/type.h"
#include "common/assert.h"
#include "dram/dram_request.h"
#include "dram/pim_request.h"
#include "hardware/hardware_config.h"

namespace llm_system {
namespace PIM_KERNEL {

template <typename... Args>
std::vector<std::function<Args...>> _PIM_KERNEL;
// auto kernel = _PIM_KERNEL<void(DRAMRequest, PIMRequest&, const
// pinballConfig)>;

using pim_kernel_ptr = void(PIMRequest&, DRAMRequestType,
                            DRAMRequest::PIM_Operand&, PIMHWConfig);

// using pim_kernel_ptr = DRAMInterface::pim_kernel_ptr;
//    void(PIMRequest&, DRAMRequestType,
//                              DRAMRequest::PIM_Operand&, pinballConfig);
//  //auto kernel = _PIM_KERNEL<pim_kernel_ptr>;

// auto kernel = _PIM_KERNEL<void(PIMRequest&, DRAMRequestType,
//   DRAMRequest::PIM_Operand&, pinballConfig)>;

// std::vector<std::function<void(DRAMRequest, PIMRequest&)>> PIM_KERNEL;

std::vector<MemoryObject::Ptr> get_operand(
    const DRAMRequest::PIM_Operand& operand, const PIMOperandType type);

void init(std::vector<std::function<pim_kernel_ptr>>& kernel);

pim_kernel_ptr Read_kernel;
pim_kernel_ptr Write_kernel;
pim_kernel_ptr Move_kernel;
pim_kernel_ptr Mult_kernel;
pim_kernel_ptr Add_kernel;
pim_kernel_ptr MAD_kernel;
pim_kernel_ptr PMult_kernel;
pim_kernel_ptr CMult_kernel;
pim_kernel_ptr CAdd_kernel;
pim_kernel_ptr CMAD_kernel;
pim_kernel_ptr Tensor_kernel;
pim_kernel_ptr Tensor_Square_kernel;
pim_kernel_ptr ModUp_Evkmult_kernel;
pim_kernel_ptr ModDownEpilogue_kernel;
pim_kernel_ptr PMult_Accum_kernel;
pim_kernel_ptr CMult_Accum_kernel;
pim_kernel_ptr GEMV;

}  // namespace PIM_KERNEL

}  // namespace llm_system
