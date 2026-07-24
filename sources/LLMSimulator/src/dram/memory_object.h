#pragma once

#include <memory>
#include <vector>

#include "base/type.h"
#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/base.h"

namespace llm_system {

// input address vector for ramulator
struct AddrVec {
  int cube;
  int channel;
  int rank;
  int bankgroup;
  int bank;
  int row;
  int col;
};

class MMapController;

class MemoryObject {
 public:
  using Ptr = std::shared_ptr<MemoryObject>;
  using MMapController_Ptr = std::shared_ptr<MMapController>;

  [[nodiscard]] static Ptr Create(MMap mmap, addr address, long long size,
                                  MMapController_Ptr mmap_controller) {
    return Ptr(new MemoryObject(mmap, address, size, mmap_controller));
  };

  Ramulator::AddrVec_t getAddrVec(long long bundle_idx,
                                  ProcessorType type = ProcessorType::GPU);
  long getBundleSize() { return num_bundle; }

  void setSize(long long _size);

 private:
  MMap mmap;
  addr address;
  long long size;
  long long num_bundle;

  MMapController_Ptr mmap_controller;

  MemoryObject() = default;
  MemoryObject(MMap mmap, addr address, long long size,
               MMapController_Ptr mmap_controller);
};

}  // namespace llm_system
