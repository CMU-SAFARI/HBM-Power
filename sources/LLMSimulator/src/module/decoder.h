#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/layernorm.h"
#include "module/residual.h"
#include "module/parallel.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class Decoder : public Module {
 public:
  using Ptr = std::shared_ptr<Decoder>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Decoder(prefix, name, model_config, scheduler,
                              device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Decoder(std::string& prefix, std::string& name,
          const ModelConfig& model_config, Scheduler::Ptr scheduler,
          std::vector<int>& device_list, Device::Ptr device);
  Decoder() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MoEDecoder : public Module {
 public:
  using Ptr = std::shared_ptr<MoEDecoder>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEDecoder(prefix, name, model_config, scheduler,
                                 device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEDecoder(std::string& prefix, std::string& name,
             const ModelConfig& model_config, Scheduler::Ptr scheduler,
             std::vector<int>& device_list, Device::Ptr device);
  MoEDecoder() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system