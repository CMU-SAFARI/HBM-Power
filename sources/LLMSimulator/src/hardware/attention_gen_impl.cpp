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

ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

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

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = 1.0 * (m * k * num_heads / num_kv_heads + k * n + m * n * num_heads / num_kv_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    accumul_len += n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  // Softmax //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = 1.0 * (m * k * num_heads / num_kv_heads + k * n + m * n * num_heads / num_kv_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    accumul_len += k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }  
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  double opb = total_flops / total_memory_size;
  exec_status.total_duration += accumul_memory_duration * opb;

  // Softmax //

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else{
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  opb = total_flops / total_memory_size;
  exec_status.total_duration += accumul_memory_duration * opb;

  // exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<int> orig_shape = input->shape;
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops =
    config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;


      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  // exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      // k_cache->setShape(shape);
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
      
      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{  
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
    
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    
      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

}  // namespace llm_system