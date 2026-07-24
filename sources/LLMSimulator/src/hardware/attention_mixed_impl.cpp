#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus AttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  return exec_status;
};

ExecStatus AttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator) {
  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  ExecStatus exec_status;

  return exec_status;
};

ExecStatus AttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus MultiLatentAttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  return exec_status;
};

// need to fix
ExecStatus MultiLatentAttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator) {
  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  ExecStatus exec_status;

  return exec_status;
};

ExecStatus MultiLatentAttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];
    int batch_size = sequences_metadata->get_seq().size();

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};


}  // namespace llm_system