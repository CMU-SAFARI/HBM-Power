#pragma once
#include <cassert>
#include <iostream>
#include <string>

#include "module/module.h"
#include "module/tensor.h"

namespace llm_system {

class SiluAndMul : public Module {
  // models suc has Mixtral, Llama, and Qwen use silu activation
 public:
  using Ptr = std::shared_ptr<SiluAndMul>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  Device::Ptr device) {
    return Ptr(new SiluAndMul(prefix, name, device));
  };

 private:
  SiluAndMul(std::string& prefix, std::string& name, Device::Ptr device);
  SiluAndMul(){};

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int activation_factor;
};

class Activation : public Module {
 public:
  using Ptr = std::shared_ptr<Activation>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int activation_factor, Device::Ptr device) {
    return Ptr(new Activation(prefix, name, activation_factor, device));
  };

 private:
  Activation(std::string& prefix, std::string& name, int activation_factor,
             Device::Ptr device);
  Activation(){};

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int activation_factor;
};

}  // namespace llm_system