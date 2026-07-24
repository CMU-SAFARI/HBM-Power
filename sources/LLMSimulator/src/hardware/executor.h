#pragma once
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_type.h"
#include "hardware/base.h"
#include "hardware/layer_impl.h"
#include "module/status.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Executor {
 public:
  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool)>>
      linear_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool, bool)>>
      batched_linear_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool)>>
      activation_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_sum_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_mixed_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_sum_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_mixed_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      absorb_mla_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      absorb_mla_sum_function_ramulator_ptr;

  void execution(LayerType layer_type,
                 const std::vector<Tensor_Ptr>& tensor_list,
                 const BatchedSequence::Ptr sequences_metadata,
                 const std::vector<ProcessorType>& processor_type,
                 const LayerInfo layer_info, bool use_ramualtor,
                 Device_Ptr device);

  Executor();

 private:
  void init();
  void initLinear();
  void initActivation();
  void initAttentionGen();
  void initAttentionSum();
  void initAttentionMixed();
  void initMultiLatentAttentionGen();
  void initMultiLatentAttentionSum();
  void initMultiLatentAttentionMixed();
  void initAbsorbMLAGen();
  void initAbsorbMLASum();

  ExecStatus executePType(LayerType layer_type,
                          const std::vector<Tensor_Ptr>& tensor_list,
                          const BatchedSequence::Ptr sequences_metadata,
                          ProcessorType processor_type,
                          const LayerInfo layer_info, bool use_ramulator,
                          Device_Ptr device);

  ExecStatus executeLinear(const std::vector<Tensor_Ptr>& tensor_list,
                           ProcessorType processor_type, bool use_ramulator,
                           Device_Ptr device);

  ExecStatus executeBatchedLinear(const std::vector<Tensor_Ptr>& tensor_list,
                           ProcessorType processor_type, bool use_ramulator,
                           bool duplicated_input, Device_Ptr device);

  ExecStatus executeActivation(const std::vector<Tensor_Ptr>& tensor_list,
                               ProcessorType processor_type, bool use_ramulator,
                               Device_Ptr device);

  ExecStatus executeAttentionGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  ExecStatus executeAttentionSum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  ExecStatus executeAttentionMixed(
      const std::vector<Tensor_Ptr>& tensor_list,
      const BatchedSequence::Ptr sequences_metadata,
      ProcessorType processor_type, const LayerInfo layer_info,
      bool use_ramulator, Device_Ptr device);
  
  ExecStatus executeMultiLatentAttentionGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  ExecStatus executeMultiLatentAttentionSum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  ExecStatus executeMultiLatentAttentionMixed(
      const std::vector<Tensor_Ptr>& tensor_list,
      const BatchedSequence::Ptr sequences_metadata,
      ProcessorType processor_type, const LayerInfo layer_info,
      bool use_ramulator, Device_Ptr device);
  
  ExecStatus executeAbsorbMLAGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  ExecStatus executeAbsorbMLASum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
                                 
};

};  // namespace llm_system