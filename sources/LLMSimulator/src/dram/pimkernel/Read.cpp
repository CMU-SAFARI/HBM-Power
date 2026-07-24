#include "dram/dram_interface.h"
#include "dram/pimkernel/pim_kernel.h"

namespace llm_system {
namespace PIM_KERNEL {

void Read_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                 DRAMRequest::PIM_Operand& operand,
                 const PIMHWConfig pim_hw_config) {
  auto read_operand = get_operand(operand, PIMOperandType::kDRAM);
  assertTrue((read_operand.size() == 1), "only one request can be read");

  Ramulator::AddrVec_t addr_vec;

  // std::cout << "size: " << std::to_string(read_operand[0]->getBundleSize())
  //           << std::endl;
  for (auto opnd : read_operand) {
    for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx, pim_hw_config.type);
        if (addr_vec.at(0) == 0) {
          pim_request.AddCommand(PIMCommand(PIMCommandType::kRead,
                                            PIMOperandType::kDRAM, addr_vec,
                                            &pim_request, dramreq_type));
        }
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system