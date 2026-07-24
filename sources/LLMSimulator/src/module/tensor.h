#pragma once
#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include "common/assert.h"
#include "dram/memory_object.h"
#include "hardware/device.h"
#include "scheduler/sequence.h"

class Module;
namespace llm_system {

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  using Ptr = std::shared_ptr<Tensor>;
  using Module_ptr = std::shared_ptr<Module>;

  [[nodiscard]] static Ptr Create(std::string name = "dummy",
                                  std::vector<int> shape = {1, 1},
                                  std::string tag = "act",
                                  Device::Ptr device = NULL,
                                  int precision_byte = 1,
                                  MMap mmap = MMap::ALL_CHANNEL) {
    Ptr ptr = Ptr(new Tensor(name, shape, tag, device, precision_byte, mmap));
    ptr->setMemoryObject();
    return ptr;
  }

  std::string name;
  std::vector<int> shape;
  std::string tag;
  int precision_byte;
  bool ready;

  // for MoE layers
  bool perform_with_optimal;
  
  // for parallel execution
  bool parallel_execution;
  bool perform_at_high;

  bool isDecoder = false;
  bool isMemPoolLoad = false;

  int dim(int i);

  void set();
  void unset();

  long getSize();

  std::vector<int>& getShape() { return shape; }

  void set_device(Device::Ptr device);
  Device::Ptr get_device() { return device; }

  std::string ToString();
  std::string get_module_map_name();

  void setPerformHigh() {
    parallel_execution = true;
    perform_at_high = true;
  }
  void setPerformLow() {
    parallel_execution = true;
    perform_at_high = false;
  }

  bool isPerformHigh() { return perform_at_high; }

  Ptr getPtr() { return shared_from_this(); }

  MMap getMMap() { return mmap; }
  MemoryObject::Ptr getMemoryObject() { return memory_object; }
  void set_module(Module_ptr _module) { module = _module; }

  bool timeboard_synced;

  void setMemoryObject();

  // set shape and update memory object size
  void setShape(std::vector<int> shape_) {
    shape = shape_;
    memory_object->setSize(getSize());
  }

  void allocateMemoryObject(MemoryObject::Ptr memory_object_) {
    memory_object = memory_object_;
  };

 private:
  Device::Ptr device;
  Module_ptr module;
  MMap mmap;
  MemoryObject::Ptr memory_object;
  Tensor(std::string name, std::vector<int> shape, std::string tag,
         Device::Ptr device, int precision_byte, MMap mmap);
};

}  // namespace llm_system