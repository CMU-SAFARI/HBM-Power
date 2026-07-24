#pragma once

#include <memory>
#include <vector>

#include "common/assert.h"
#include "common/type.h"
#include "dram/dram_type.h"

namespace llm_system {

/**
 * @brief
 */

class DataObject {
 public:
  using Ptr = std::shared_ptr<DataObject>;

  static Ptr Create(MMap mmap, MemoryObject) {
    return DataObject::Ptr(new DataObject(mmap, address));
  }
  static Ptr Create(const Address &addr) { return Ptr(new DataObject(addr)); }

  Ptr DeepCopy() { return Create(addr); }

  // Enable moving, but not copying
  DataObject(DataObject &&) = default;
  DataObject &operator=(DataObject &&) = default;

  const Address &GetAddress() const { return addr; }

  const int GetRowAddr(int offset);
  const int GetColAddr(int offset);
  ~DataObject() = default;

 private:
  DataObject() = default;
  DataObject(const Address &addr) : addr{addr} {}
  DataObject(MMap mmap, Address address);
  MMap mmap;
  Address addr;
};

}  // namespace llm_system
