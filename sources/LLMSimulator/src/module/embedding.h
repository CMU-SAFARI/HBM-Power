#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Embedding : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<Embedding>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new Embedding(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
    // return Ptr(new Embedding(name, input_size, output_size,
    //                                     device_list, device));
  };

 private:
  Embedding(std::string& prefix, std::string& name, ModelConfig model_config,
            std::vector<int> device_list, Device::Ptr device);
  Embedding() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

}  // namespace llm_system