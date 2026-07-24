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

class Linear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<Linear>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new Linear(prefix, name, input_size, output_size, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Linear(std::string& prefix, std::string& name, int input_size,
         int output_size, std::vector<int> device_list, Device::Ptr device);
  Linear() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class BatchedLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<BatchedLinear>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int num_batched_gemm, int input_size, int output_size,
                                  bool duplicated_input, bool use_plain_lienar, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new BatchedLinear(prefix, name, num_batched_gemm, input_size, output_size, duplicated_input, use_plain_lienar, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  BatchedLinear(std::string& prefix, std::string& name, int num_batched_gemm, int input_size,
         int output_size, bool duplicated_input, bool use_plain_lienar, std::vector<int> device_list, Device::Ptr device);
  BatchedLinear() = default;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool duplicated_input;
  bool use_plain_linear;
};

}  // namespace llm_system