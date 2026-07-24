#pragma once

#include <list>
#include <memory>
#include <vector>

#include "base/type.h"
#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_object.h"

namespace llm_system {

class PIMRequest;

class PIMCommand {
 public:
  DRAMRequestType dramreq_type;
  PIMCommandType pimcmd_type;
  PIMOperandType op_type;
  Ramulator::AddrVec_t addr_vec;

  PIMRequest* request = NULL;

  PIMCommand() = default;

  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec);
  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest);
  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest,
             DRAMRequestType dramreq_type);
};

class PIMRequest {
 public:
  friend class DRAMInterface;
  PIMRequest();

  void init();
  std::list<PIMCommand> command_queue;

  // stat
  std::vector<counter_t> issued_pim_cmd;
  std::vector<counter_t> issued_dram_cmd;

  cycle_t start = 0;
  cycle_t end = 0;

  void AddCommand(PIMCommand&& pimcommand);
  std::list<PIMCommand>& GetCommand();
};

}  // namespace llm_system
