#include "dram/memory_object.h"

#include "common/assert.h"
#include "common/type.h"
#include "dram/mmap_controller.h"

namespace llm_system {

MemoryObject::MemoryObject(MMap mmap, addr address, long long size,
                           MMapController_Ptr mmap_controller)
    : mmap(mmap),
      address(address),
      size(size),
      mmap_controller(mmap_controller) {
  num_bundle = size / mmap_controller->getGranul();
}

Ramulator::AddrVec_t MemoryObject::getAddrVec(long long bundle_idx,
                                              ProcessorType type) {
  Ramulator::AddrVec_t addrvec;
  if (type == ProcessorType::LOGIC) {
    addrvec = mmap_controller->getAddrVecLOGIC(address, bundle_idx);
  }
  else {
    addrvec = mmap_controller->getAddrVec(address, bundle_idx);
  }

  return addrvec;
}

void MemoryObject::setSize(long long size_) {
  size = size_;
  num_bundle = size / mmap_controller->getGranul();
}

}  // namespace llm_system
