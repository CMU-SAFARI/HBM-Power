#include "dram/mmap_controller.h"

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

MMapController::MMapController(MemoryConfig memory_config)
    : memory_config(memory_config) {
  start_addr_normal = 0;
  start_addr_logic = 0;
};

void MMapController::setMemoryObject(Tensor::Ptr tensor) {
  MMap map = tensor->getMMap();
  if (map == MMap::ALL_CHANNEL) {
    setNormal(tensor);
  }
}

Ramulator::AddrVec_t MMapController::getAddrVec(addr address, long long bundle_idx) {
  addr target_address = address + bundle_idx * memory_config.granul;
  AddrVec addr_vec = addrToVec(target_address);
  int channel = memory_config.num_channel * addr_vec.cube + addr_vec.channel;
  Ramulator::AddrVec_t ramul_addr_vec = {
      channel / 2,   channel % 2,  addr_vec.rank, addr_vec.bankgroup,
      addr_vec.bank, addr_vec.row, addr_vec.col};

  return ramul_addr_vec;
}

Ramulator::AddrVec_t MMapController::getAddrVecLOGIC(addr address,
                                                     long long bundle_idx) {
  addr target_address = address + bundle_idx * memory_config.granul;
  AddrVec addr_vec = addrToVecLOGIC(target_address);
  int channel = memory_config.num_channel * addr_vec.cube + addr_vec.channel;
  Ramulator::AddrVec_t ramul_addr_vec = {
      channel / 2,   channel % 2,  addr_vec.rank, addr_vec.bankgroup,
      addr_vec.bank, addr_vec.row, addr_vec.col};

  return ramul_addr_vec;
}

AddrVec MMapController::addrToVec(addr address) {
  assertTrue(address >= 0, "Address is not valid");
  long long idx = 0;

  address /= memory_config.granul;

  int cube_idx = address % memory_config.num_cube;
  address /= memory_config.num_cube;

  int channel_idx = address % memory_config.num_channel;
  address /= memory_config.num_channel;

  int rank_idx = address % memory_config.num_rank;
  address /= memory_config.num_rank;

  int bg_idx = address % memory_config.num_bankgroup;
  address /= memory_config.num_bankgroup;

  int bank_idx = address % memory_config.num_bank;
  address /= memory_config.num_bank;

  int col_idx = address % memory_config.num_col;
  address /= memory_config.num_col;

  int row_idx = address;
  assertTrue(row_idx < memory_config.num_row, "Unvalid memory hasing");

  AddrVec addr = {cube_idx, channel_idx, rank_idx, bg_idx,
                  bank_idx, row_idx,     col_idx};
  return addr;
}

AddrVec MMapController::addrToVecLOGIC(addr address) {
  long long idx = 0;

  address /= memory_config.granul;

  int cube_idx = address % memory_config.num_logic_cube;
  address /= memory_config.num_logic_cube;

  int channel_idx = address % memory_config.num_channel;
  address /= memory_config.num_channel;

  // PIM memory set
  int bg_idx = address % memory_config.num_bankgroup;
  address /= memory_config.num_bankgroup;

  int bank_under_idx = address % 2;
  address /= 2;

  // PIM memory set

  int col_idx = address % memory_config.num_col;
  address /= memory_config.num_col;

  int row_idx = address % memory_config.num_row;
  address /= memory_config.num_row;

  int bank_upper_idx = address % 2;
  address /= 2;

  int rank_idx = address % memory_config.num_rank;

  int bank_idx = bank_upper_idx * 2 + bank_under_idx;

  assertTrue(rank_idx < memory_config.num_rank, "Unvalid memory hasing");

  AddrVec addr = {cube_idx, channel_idx, rank_idx, bg_idx,
                  bank_idx, row_idx,     col_idx};
  return addr;
}

addr MMapController::vecToAddrLOGIC(AddrVec addr_vec) {
  addr address = 0;
  int cube_idx = addr_vec.cube;
  int channel_idx = addr_vec.channel;
  int rank_idx = addr_vec.rank;
  int bankgroup_idx = addr_vec.bankgroup;
  int bank_idx = addr_vec.bank;
  int row_idx = addr_vec.row;
  int col_idx = addr_vec.col;

  int bank_upper_idx = bank_idx / 2;
  int bank_under_idx = bank_idx % 2;

  address += rank_idx;

  address *= 2;
  address += bank_upper_idx;

  address *= memory_config.num_row;
  address += row_idx;

  address *= memory_config.num_col;
  address += col_idx;

  address *= 2;
  address += bank_under_idx;

  address *= memory_config.num_bankgroup;
  address += bankgroup_idx;

  address *= memory_config.num_channel;
  address += channel_idx;

  address *= memory_config.num_cube;
  address += cube_idx;

  address *= memory_config.granul;

  return address;
}

// allocate tensor in all channel
void MMapController::setNormal(Tensor::Ptr tensor) {
  addr start_address = start_addr_normal;
  if (tensor->tag.compare("act") && tensor->tag.compare("cache")) {
    start_addr_normal += tensor->getSize();
  } else {
    start_address = 0;
  }

  MemoryObject::Ptr memory_object = MemoryObject::Create(
      tensor->getMMap(), start_address, tensor->getSize(), getPtr());
  tensor->allocateMemoryObject(memory_object);
}

// allocate tensor in all channel
void MMapController::setLOGIC(Tensor::Ptr tensor) {
  MemoryObject::Ptr memory_object = MemoryObject::Create(
      tensor->getMMap(), start_addr_logic, tensor->getSize(), getPtr());
  tensor->allocateMemoryObject(memory_object);
  start_addr_logic += tensor->getSize();
}

}  // namespace llm_system
