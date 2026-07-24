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

ExecStatus AttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);
  // Tensor_Ptr k_cache = tensor.at(1);
  //  Tensor_Ptr v_cache = tensor.at(2);

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

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size =
        1.0 * (m * k * num_heads + k * n * num_kv_heads + m * n * num_heads) *
        input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // read query
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      // read value
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;

      // write score
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // read query
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device,ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      // read value
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask//
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    memory_size = 1.0 * (m * n * num_heads * 2) * input->precision_byte;
    memory_size = 0;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size =
        1.0 * (m * k * num_heads + k * n * num_kv_heads + m * n * num_heads) *
        input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      // read score
      auto shape = input->getShape();

      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;

      // read kv

      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      // read score
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read kv
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

// need to fix
ExecStatus AttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
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
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);

      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionSumExecutionPIM(Device_Ptr device,
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
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = m;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = m;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = k;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = k;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

// Multi Latent Attention // 
ExecStatus MultiLatentAttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = std::min((shared_mem_size / (4 * k * input->precision_byte)), static_cast<double>(k)); // num_rows, B_r
      int block_size_c = shared_mem_size / (4 * k * input->precision_byte);

      int num_tile_r = (m + block_size_r - 1)/ block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);

      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +   // query
                     1.0 * (k_i + v_i) * num_tile_c +        // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +  // output, read/write output 
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c) *
                     num_heads * input->precision_byte;      // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000; 
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv

        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

// Multi Latent Attention // 
ExecStatus MultiLatentAttentionSumExecutionLogic(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = shared_mem_size / (4 * k * input->precision_byte); // num_rows
      int block_size_c = k;  // num_rows

      int num_tile_r = (m + block_size_r - 1) / block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);
      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +        // query
                     1.0 * (k_i + v_i) * num_tile_c +             // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +        // output, read/write output
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c  // read/write norm_stat
                    ) * num_heads * input->precision_byte;        // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
          // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv

        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  
  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

ExecStatus MultiLatentAttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;


  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = shared_mem_size / (4 * k * input->precision_byte); // num_rows
      int block_size_c = k;  // num_rows

      int num_tile_r = (m + block_size_r - 1) / block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);
      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +        // query
                     1.0 * (k_i + v_i) * num_tile_c +             // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +        // output, read/write output
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c  // read/write norm_stat
                    ) * num_heads * input->precision_byte;        // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;
        
        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;

      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        
        ExecStatus temp;
          // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};


ExecStatus AbsorbMLASumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);
  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  
  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  int num_seq = seq_list.size();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  // Scoring for nope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000; 
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // read query x W_UK
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write score for nope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // read query x W_UK
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for nope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Scoring for rope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write score for rope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for rope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
    // memory_size = 0; // can be overlapped
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // read input
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      // store output
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // read input
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store output
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = 1.0 *
                  (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      // read score
      time_ns accumul_memory_duration = 0;

      std::vector<int> shape = {1,1};

      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv

      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // read score
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  input->setShape(orig_shape);
  
  return exec_status;
};

ExecStatus AbsorbMLASumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

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

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
      1.0 * m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Scoring for rope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;

      // write score for rope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for rope
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;
    total_memory_size += memory_size;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = 1.0 *
                  (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      // read score
      std::vector<int> shape = {1,1};

      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;

      // read compressed kv
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      // read score
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AbsorbMLASumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  // fail("not implemented");
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

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

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  time_ns total_duration = 0;

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring for NoPE//
  int num_seq = seq_list.size();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;
    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }
  
  // Scoring for RoPE // 
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n+
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (1.0 * m * k * num_heads + k * n + 1.0 * m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;
    
    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = k;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = k;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};
}  // namespace llm_system