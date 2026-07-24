#pragma once
#include <iostream>

#include "common/type.h"

namespace llm_system {

struct Stat {
  time_ns time;
  time_ns latency;
  time_ns queueing_delay = 0;
  time_ns arrival_time = 0;
  int seq_queue_size = 0;

  int process_token = 0;
  int batchsize = 0;
  int sum_seq = 0;
  int gen_seq = 0;
  int average_seq_len = 0;

  int input_len = 0;
  int output_len = 0;
  int num_sum_iter = 0;
  double sum_attention_opb = 0.0;
  int end_token = 0;

  time_ns qkv_gen = 0;
  time_ns atten_sum = 0;
  time_ns atten_gen = 0;
  time_ns o_proj = 0;
  time_ns ffn = 0;
  time_ns expert_ffn = 0;
  time_ns communication = 0;

  // for MLA //
  time_ns q_down_proj = 0;
  time_ns kv_down_proj = 0;
  time_ns kr_proj = 0;
  time_ns q_up_proj = 0;
  time_ns qr_proj = 0;

  // MLA (Base) //
  time_ns kv_up_proj = 0;
  
  // MLA (Absorb) //
  time_ns tr_k_up_proj = 0;
  time_ns v_up_proj = 0;

  time_ns rope = 0;
  time_ns layernorm = 0;
  time_ns residual = 0;

  energy_nJ act_energy = 0;
  energy_nJ read_energy = 0;
  energy_nJ write_energy = 0;
  energy_nJ all_act_energy = 0;
  energy_nJ all_read_energy = 0;
  energy_nJ all_write_energy = 0;
  energy_nJ mac_energy = 0;
  energy_nJ total_energy = 0;

  energy_nJ FC_DRAM_energy = 0;
  energy_nJ FC_COMP_energy = 0;
  energy_nJ Attn_DRAM_energy = 0;
  energy_nJ Attn_COMP_energy = 0;
  energy_nJ MoE_DRAM_energy = 0;
  energy_nJ MoE_COMP_energy = 0;

  bool isOOM = false;

  int is_mixed = 0; // wheter it is mixed stage or not
  int split = 0;
  std::string type;  // t2t, t2ft, e2e
  int iter_info;
};

}  // namespace llm_system