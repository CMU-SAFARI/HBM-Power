#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/parallel.h"
#include "module/compressed_kv_restore.h"
#include "module/layernorm.h"
#include "module/rope.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class Attention : public Module {
 public:
  using Ptr = std::shared_ptr<Attention>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Attention(prefix, name, model_config, scheduler,
                                device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Attention(std::string& prefix, std::string& name,
            const ModelConfig& model_config, Scheduler::Ptr scheduler,
            std::vector<int>& device_list, Device::Ptr device);
  Attention(){};

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MultiLatentAttention : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttention>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   const ModelConfig& model_config,
                                   Scheduler::Ptr scheduler,
                                   std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttention(prefix, name, model_config, scheduler,
                                 device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
   MultiLatentAttention(std::string& prefix, std::string& name,
             const ModelConfig& model_config, Scheduler::Ptr scheduler,
             std::vector<int>& device_list, Device::Ptr device);
   MultiLatentAttention(){};
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  
   bool use_absorb;
   int num_heads;
   int parallel_num;
 };

class FeedForward2Way : public Module {
 public:
  using Ptr = std::shared_ptr<FeedForward2Way>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device,
                                  bool perform_all_reduce = true,
                                  bool is_expert = false,
                                  bool use_dp = false) {
    Ptr ptr = Ptr(new FeedForward2Way(prefix, name, model_config, scheduler,
                                      device_list, device, perform_all_reduce,
                                      is_expert, use_dp));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  FeedForward2Way(std::string& prefix, std::string& name,
                  const ModelConfig& model_config, Scheduler::Ptr scheduler,
                  std::vector<int>& device_list, Device::Ptr device,
                  bool perform_all_reduce, bool is_expert, bool use_dp);
  FeedForward2Way() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool perform_all_reduce;
};

class FeedForward3Way : public Module {
 public:
  using Ptr = std::shared_ptr<FeedForward3Way>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device,
                                  bool perform_all_reduce = true,
                                  bool is_expert = false,
                                  bool use_dp = false) {
    Ptr ptr = Ptr(new FeedForward3Way(prefix, name, model_config, scheduler,
                                      device_list, device, perform_all_reduce,
                                      is_expert, use_dp));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  FeedForward3Way(std::string& prefix, std::string& name,
                  const ModelConfig& model_config, Scheduler::Ptr scheduler,
                  std::vector<int>& device_list, Device::Ptr device,
                  bool perform_all_reduce, bool is_expert, bool use_dp);
  FeedForward3Way() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool perform_all_reduce;
};

}  // namespace llm_system