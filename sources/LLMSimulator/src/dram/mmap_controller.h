#pragma once

#include <memory>
#include <vector>

#include "base/type.h"
#include "common/assert.h"
#include "common/type.h"
#include "dram/memory_config.h"
#include "dram/memory_object.h"
#include "module/tensor.h"

namespace llm_system {

class MMapController : public std::enable_shared_from_this<MMapController> {
 public:
  using Ptr = std::shared_ptr<MMapController>;

  static Ptr Create(MemoryConfig memory_config) {
    return MMapController::Ptr(new MMapController(memory_config));
  }

  void setMemoryObject(Tensor::Ptr tensor);

  Ptr getPtr() { return shared_from_this(); }
  Ramulator::AddrVec_t getAddrVec(addr address, long long bundle_idx);
  Ramulator::AddrVec_t getAddrVecLOGIC(addr address, long long bundle_idx);

  int getGranul() { return memory_config.granul; }
  MemoryConfig getConfig(){ return memory_config; }

 private:
  addr start_addr_normal;
  addr start_addr_logic;

  MMapController(MemoryConfig memory_config);

   // channel = pseudochannel in this function
   void setNormal(Tensor::Ptr tensor);   // allocate to all channel
   void setLOGIC(Tensor::Ptr tensor);    // allocate to channel in a cube

   // not used
   void setChannel(Tensor::Ptr tensor);  // allocate to one channel

   AddrVec addrToVec(addr addr);
   
   AddrVec addrToVecLOGIC(addr addr);

   addr vecToAddrLOGIC(AddrVec addr_vec);

   MemoryConfig memory_config;
};

}  // namespace llm_system
