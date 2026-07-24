#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/pimkernel/pim_kernel.h"

namespace llm_system {
namespace PIM_KERNEL {

void GEMV(PIMRequest& pim_request, DRAMRequestType dramreq_type,
          DRAMRequest::PIM_Operand& operand, const PIMHWConfig pim_hw_config) {
  const int bandwidth_x = pim_hw_config.bandwidth_x;
  const ProcessorType type = pim_hw_config.type;
  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0, 0};

  auto src = get_operand(operand, PIMOperandType::kSrc);
  // auto dest = get_operand(operand, PIMOperandType::kDest);

  assertTrue(src.size() == 1, "GEMV operands are not valid");

  MemoryObject::Ptr opnd = src.at(0);

  if (type == ProcessorType::PIM) {
    if (bandwidth_x == 16) {
      for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize();
           bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx);
        if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
            addr_vec.at(3) == 0 && addr_vec.at(4) == 0) {
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    } else if (bandwidth_x == 8) {
      for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize();
           bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx);
        if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
            addr_vec.at(3) == 0 && (addr_vec.at(4) % 2 == 0)) {
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    } else if (bandwidth_x == 4) {
      for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize();
           bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx);
        // if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 &&
        //     (addr_vec.at(4) % 2 == 0) && addr_vec.at(3) == 0) {
        if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
            addr_vec.at(3) == 0) {
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    } else {
      fail("Not supported bandwidth_x configuration");
    }
  } else if (type == ProcessorType::LOGIC) {
    if (bandwidth_x == 4) {
      for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize();
           bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx, type);
        if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 &&
            (addr_vec.at(4) % 2 == 0) && addr_vec.at(3) == 0) {
          // if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) ==
          // 0 &&
          //     addr_vec.at(3) == 0) {
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    } else {
      fail("Not supported bandwidth_x configuration");
    }
  }
}


// void GEMV(PIMRequest& pim_request, DRAMRequestType dramreq_type,
//           DRAMRequest::PIM_Operand& operand, const PIMHWConfig pim_hw_config) {
//   const int bandwidth_x = pim_hw_config.bandwidth_x;
//   Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0, 0};

//   auto src = get_operand(operand, PIMOperandType::kSrc);
//   // auto dest = get_operand(operand, PIMOperandType::kDest);

//   assertTrue(src.size() == 1, "GEMV operands are not valid");

//   MemoryObject::Ptr opnd = src.at(0);

//   if (bandwidth_x == 16) {
//     for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
//       addr_vec = opnd->getAddrVec(bundle_idx);
//       if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
//           addr_vec.at(3) == 0 && addr_vec.at(4) == 0) {
//         pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
//                                           PIMOperandType::kSrc, addr_vec,
//                                           &pim_request, dramreq_type));
//       }
//     }
//   } else if (bandwidth_x == 8) {
//     for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
//       addr_vec = opnd->getAddrVec(bundle_idx);
//       if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
//           addr_vec.at(3) == 0 && (addr_vec.at(4) % 2 == 0)) {
//         pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
//                                           PIMOperandType::kSrc, addr_vec,
//                                           &pim_request, dramreq_type));
//       }
//     }
//   } else if (bandwidth_x == 4) {
//     for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
//       addr_vec = opnd->getAddrVec(bundle_idx);
//       if (addr_vec.at(0) == 0 && addr_vec.at(1) == 0 && addr_vec.at(2) == 0 &&
//           addr_vec.at(3) == 0) {
//         pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
//                                           PIMOperandType::kSrc, addr_vec,
//                                           &pim_request, dramreq_type));
//       }
//     }
//   } else {
//     fail("Not supported bandwidth_x configuration");
//   }
// }

}  // namespace PIM_KERNEL
}  // namespace llm_system