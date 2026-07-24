#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/parallel.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class ExpertFFN : public Module {
 public:
  using Ptr = std::shared_ptr<ExpertFFN>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ExpertFFN(prefix, name, model_config, scheduler,
                                device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  ExpertFFN(std::string& prefix, std::string& name,
            const ModelConfig& model_config, Scheduler::Ptr scheduler,
            std::vector<int>& device_list, Device::Ptr device);
  ExpertFFN() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int expert_offset;
  int num_expert_per_device;
  int num_shared_expert;
  std::string expert_file_path;
};

}  // namespace llm_system