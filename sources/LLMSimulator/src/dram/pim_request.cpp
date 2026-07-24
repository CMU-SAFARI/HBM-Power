#include "dram/pim_request.h"

#include "common/assert.h"

namespace llm_system {

PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec)
    : pimcmd_type(pimcmd_type), op_type(op_type), addr_vec(addr_vec) {}

PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest)
    : pimcmd_type(pimcmd_type),
      op_type(op_type),
      addr_vec(addr_vec),
      request(pimrequest) {}

PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest,
                       DRAMRequestType dramreq_type)
    : pimcmd_type(pimcmd_type),
      op_type(op_type),
      addr_vec(addr_vec),
      dramreq_type(dramreq_type),
      request(pimrequest) {}

PIMRequest::PIMRequest() { init(); }

void PIMRequest::init() {
  issued_pim_cmd.resize((int)PIMCommandType::kMAX, 0);
  issued_dram_cmd.resize((int)DRAMCommandType::kMAX, 0);
}

void PIMRequest::AddCommand(PIMCommand&& pimcommand) {
  command_queue.push_back(pimcommand);

  // update the number of issued pimcmd
  issued_pim_cmd[(int)pimcommand.pimcmd_type]++;
}

std::list<PIMCommand>& PIMRequest::GetCommand() { return command_queue; }

}  // namespace llm_system