#include "module/tensor.h"

#include "common/assert.h"
#include "module/module.h"

namespace llm_system {
Tensor::Tensor(std::string name, std::vector<int> shape, std::string tag,
               Device::Ptr device, int precision_byte, MMap mmap)
    : name(name),
      shape(shape),
      tag(tag),
      device(device),
      precision_byte(precision_byte),
      mmap(mmap) {
  ready = false;
  timeboard_synced = false;

  perform_with_optimal = false;
  parallel_execution = false;
  perform_at_high = true;
}

long Tensor::getSize() {
  long size = precision_byte;
  for (int dim : shape) {
    size *= dim;
  }
  return size;
}

int Tensor::dim(int i) {
  assert(i < shape.size());
  return shape[i];
}

void Tensor::set() {
  ready = true;
  memory_object->setSize(getSize());
}

void Tensor::unset() { ready = false; }

void Tensor::setMemoryObject() {
  assertFalse(device == nullptr, "Tensor is not allocated in device");
  device->setMemoryObject(getPtr());
}

void Tensor::set_device(Device::Ptr _device) { device = _device; }

std::string Tensor::get_module_map_name() { return module->module_map_name; }

std::string Tensor::ToString() {
  return "(" + name + ": " + std::to_string(shape.size()) + ", " + tag + ")";
}

}  // namespace llm_system