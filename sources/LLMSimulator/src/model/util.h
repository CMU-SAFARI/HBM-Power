#pragma once
#include <string>
#include <vector>

namespace llm_system {
void inline set_device_list(std::vector<int> &device_list, int device_offset,
                            int num_device) {
  device_list.resize(0);

  for (int device = device_offset; device < device_offset + num_device;
       device++) {
    device_list.push_back(device);
  }
}
}  // namespace llm_system