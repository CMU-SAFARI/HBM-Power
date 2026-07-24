#include "hardware/executor.h"

#include <iostream>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/device.h"
#include "module/tensor.h"

namespace llm_system {

Executor::Executor() { init(); }

void Executor::init() {
  initLinear();
  initActivation();
  initAttentionGen();
  initAttentionSum();
  initAttentionMixed();
  initMultiLatentAttentionGen();
  initMultiLatentAttentionSum();
  initMultiLatentAttentionMixed();
  initAbsorbMLAGen();
  initAbsorbMLASum();
}

void Executor::initLinear() {
  linear_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  linear_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      LinearExecutionGPU;
  linear_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      LinearExecutionLogic;
  linear_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      LinearExecutionPIM;
  
  batched_linear_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  batched_linear_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      BatchedLinearExecutionGPU;
  batched_linear_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      BatchedLinearExecutionLogic;
  batched_linear_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      BatchedLinearExecutionPIM;
}

void Executor::initActivation() {
  activation_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  activation_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      ActivationExecutionGPU;
  activation_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      ActivationExecutionLogic;
  activation_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      ActivationExecutionPIM;
}

void Executor::initAttentionGen() {
  attention_gen_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  attention_gen_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      AttentionGenExecutionGPU;
  attention_gen_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      AttentionGenExecutionLogic;
  attention_gen_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      AttentionGenExecutionPIM;
}

void Executor::initAttentionSum() {
  attention_sum_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  attention_sum_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      AttentionSumExecutionGPU;
  attention_sum_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      AttentionSumExecutionLogic;
  attention_sum_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      AttentionSumExecutionPIM;
}

void Executor::initAttentionMixed() {
  attention_mixed_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  attention_mixed_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      AttentionMixedExecutionGPU;
  attention_mixed_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      AttentionMixedExecutionLogic;
  attention_mixed_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      AttentionMixedExecutionPIM;
}
void Executor::initMultiLatentAttentionGen() {
  multi_latent_attention_gen_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  multi_latent_attention_gen_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      MultiLatentAttentionGenExecutionGPU;
  multi_latent_attention_gen_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      MultiLatentAttentionGenExecutionLogic;
  multi_latent_attention_gen_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      MultiLatentAttentionGenExecutionPIM;
}

void Executor::initMultiLatentAttentionSum() {
  multi_latent_attention_sum_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  multi_latent_attention_sum_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      MultiLatentAttentionSumExecutionGPU;
  multi_latent_attention_sum_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      MultiLatentAttentionSumExecutionLogic;
  multi_latent_attention_sum_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      MultiLatentAttentionSumExecutionPIM;
}

void Executor::initMultiLatentAttentionMixed() {
  multi_latent_attention_mixed_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  multi_latent_attention_mixed_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      MultiLatentAttentionMixedExecutionGPU;
  multi_latent_attention_mixed_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      MultiLatentAttentionMixedExecutionLogic;
  multi_latent_attention_mixed_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      MultiLatentAttentionMixedExecutionPIM;
}

void Executor::initAbsorbMLAGen() {
  absorb_mla_gen_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  absorb_mla_gen_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      AbsorbMLAGenExecutionGPU;
  absorb_mla_gen_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      AbsorbMLAGenExecutionLogic;
  absorb_mla_gen_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      AbsorbMLAGenExecutionPIM;
}

void Executor::initAbsorbMLASum() {
  absorb_mla_sum_function_ramulator_ptr.resize((int)ProcessorType::MAX);
  absorb_mla_sum_function_ramulator_ptr.at((int)ProcessorType::GPU) =
      AbsorbMLASumExecutionGPU;
  absorb_mla_sum_function_ramulator_ptr.at((int)ProcessorType::LOGIC) =
      AbsorbMLASumExecutionLogic;
  absorb_mla_sum_function_ramulator_ptr.at((int)ProcessorType::PIM) =
      AbsorbMLASumExecutionPIM;
}

void Executor::execution(LayerType layer_type,
                         const std::vector<Tensor_Ptr>& tensor_list,
                         const BatchedSequence::Ptr sequences_metadata,
                         const std::vector<ProcessorType>& processor_type,
                         const LayerInfo layer_info, bool use_ramualtor,
                         Device::Ptr device) {
  ExecStatus optimal_status, status;
  time_ns duration = 0;

  // processor_type is not used, use processor_type in layer_info
  for (auto type : layer_info.processor_type) {
    status = executePType(layer_type, tensor_list, sequences_metadata, type,
                          layer_info, use_ramualtor, device);
    if (duration == 0 || (duration > status.total_duration)) {
      optimal_status = status;
      duration = status.total_duration;
    }
  }
  optimal_status.parallel_execution = layer_info.parallel_execution;
  device->setExecStatus(optimal_status);
};

ExecStatus Executor::executePType(LayerType layer_type,
                                  const std::vector<Tensor_Ptr>& tensor_list,
                                  const BatchedSequence::Ptr sequences_metadata,
                                  ProcessorType processor_type,
                                  const LayerInfo layer_info,
                                  bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;
  int bandwidth_x = 0;

  if (processor_type == ProcessorType::LOGIC) {
    bandwidth_x = device->config.logic_x;
  } else if (processor_type == ProcessorType::PIM) {
    bandwidth_x = device->config.pim_x;
  }

  device->dram_interface->setPIMHWConfig(processor_type, bandwidth_x);

  if (layer_type == LayerType::LINEAR) {
    exec_status =
        executeLinear(tensor_list, processor_type, use_ramulator, device);
  } else if (layer_type == LayerType::BATCHED_LINEAR) {
    exec_status =
        executeBatchedLinear(tensor_list, processor_type, use_ramulator, layer_info.duplicated_input, device);
  } else if (layer_type == LayerType::ACTIVATION) {
    exec_status =
        executeActivation(tensor_list, processor_type, use_ramulator, device);
  } else if (layer_type == LayerType::ATTENTION_SUM) {
    exec_status =
        executeAttentionSum(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::ATTENTION_GEN) {
    exec_status =
        executeAttentionGen(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::ATTENTION_MIXED) {
    exec_status =
        executeAttentionMixed(tensor_list, sequences_metadata, processor_type,
                              layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::MLA_SUM) {
    exec_status =
        executeMultiLatentAttentionSum(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::MLA_GEN) {
    exec_status =
        executeMultiLatentAttentionGen(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::MLA_MIXED) {
    exec_status =
        executeMultiLatentAttentionMixed(tensor_list, sequences_metadata, processor_type,
                              layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::ABSORBED_MLA_SUM) {
    exec_status =
        executeAbsorbMLASum(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  } else if (layer_type == LayerType::ABSORBED_MLA_GEN) {
    exec_status =
        executeAbsorbMLAGen(tensor_list, sequences_metadata, processor_type,
                            layer_info, use_ramulator, device);
  }

  exec_status.processor_type = processor_type;

  return exec_status;
}

ExecStatus Executor::executeLinear(const std::vector<Tensor::Ptr>& tensor_list,
                                   ProcessorType processor_type,
                                   bool use_ramulator, Device::Ptr device) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  ExecStatus exec_status;

  exec_status = linear_function_ramulator_ptr.at(int(processor_type))(
      device, input, weight, output, use_ramulator);

  return exec_status;
}

ExecStatus Executor::executeBatchedLinear(const std::vector<Tensor::Ptr>& tensor_list,
                                  ProcessorType processor_type,
                                  bool use_ramulator, bool duplicated_input, Device::Ptr device) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  ExecStatus exec_status;

  exec_status = batched_linear_function_ramulator_ptr.at(int(processor_type))(
  device, input, weight, output, use_ramulator, duplicated_input);

  return exec_status;
}

ExecStatus Executor::executeActivation(
    const std::vector<Tensor::Ptr>& tensor_list, ProcessorType processor_type,
    bool use_ramulator, Device::Ptr device) {
  Tensor::Ptr gate_output = tensor_list.at(0);
  Tensor::Ptr input = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  ExecStatus exec_status;

  exec_status = activation_function_ramulator_ptr.at(int(processor_type))(
      device, gate_output, input, output, use_ramulator);

  return exec_status;
}

ExecStatus Executor::executeAttentionGen(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = attention_gen_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}
ExecStatus Executor::executeAttentionSum(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = attention_sum_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}
ExecStatus Executor::executeAttentionMixed(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = attention_mixed_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}
ExecStatus Executor::executeMultiLatentAttentionGen(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = multi_latent_attention_gen_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}
ExecStatus Executor::executeMultiLatentAttentionSum(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = multi_latent_attention_sum_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}
ExecStatus Executor::executeMultiLatentAttentionMixed(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = multi_latent_attention_mixed_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}

ExecStatus Executor::executeAbsorbMLAGen(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = absorb_mla_gen_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}

ExecStatus Executor::executeAbsorbMLASum(
    const std::vector<Tensor_Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, ProcessorType processor_type,
    const LayerInfo layer_info, bool use_ramulator, Device_Ptr device) {
  ExecStatus exec_status;

  exec_status = absorb_mla_sum_function_ramulator_ptr.at(int(processor_type))(
      device, tensor_list, sequences_metadata, layer_info, use_ramulator);

  return exec_status;
}

};  // namespace llm_system