#pragma once
#include "hardware/cluster.h"
#include "model/model_config.h"
#include "model/util.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class LLM : public Module {
 public:
  using Ptr = std::shared_ptr<LLM>;
  [[nodiscard]] static Ptr Create(const ModelConfig& model_config,
                                  Cluster::Ptr cluster,
                                  Scheduler::Ptr scheduler,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new LLM(model_config, cluster, scheduler, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  LLM(const ModelConfig& model_config, Cluster::Ptr cluster,
      Scheduler::Ptr scheduler, Device::Ptr device);

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

}  // namespace llm_system
