#pragma once
#include <map>
#include <string>

#include "common/assert.h"

namespace llm_system {

class ModelConfig {
  // default: mixtral 8x7B
 public:
  ModelConfig(int hidden_dim = 4096, int head_dim = 128, int num_layers = 32,
              int num_heads = 32, int num_kv_heads = 8, int max_seq_len = 32768,
              int intermediate_dim = 14336, int expert_intermediate_dim = 14336,
              int activation_factor = 1, int precision_byte = 2,
              int num_routed_expert = 8, int num_shared_expert = 0,
              int expert_freq = 1, int top_k = 2, int ffn_way = 3,
              int first_k_dense = 0, int q_lora_rank = 0, int kv_lora_rank = 0,
              int qk_nope_head_dim = 0, int qk_rope_head_dim = 0, int n_vocab = 32000, bool compressed_kv = false, 
              bool use_absorb = false,
              double skewness = 0.0,              
              std::string model_name = "")
      : hidden_dim(hidden_dim),
        head_dim(head_dim),
        num_layers(num_layers),
        num_heads(num_heads),
        num_kv_heads(num_kv_heads),
        max_seq_len(max_seq_len),
        intermediate_dim(intermediate_dim),
        expert_intermediate_dim(expert_intermediate_dim),
        activation_factor(activation_factor),
        precision_byte(precision_byte),
        num_routed_expert(num_routed_expert),
        num_shared_expert(num_shared_expert),
        expert_freq(expert_freq),
        top_k(top_k),
        ffn_way(ffn_way),
        first_k_dense(first_k_dense),
        q_lora_rank(q_lora_rank),
        kv_lora_rank(kv_lora_rank),
        qk_nope_head_dim(qk_nope_head_dim),
        qk_rope_head_dim(qk_rope_head_dim),
        n_vocab(n_vocab),
        compressed_kv(compressed_kv),
        use_absorb(use_absorb),
        skewness(skewness),        
        model_name(model_name) {
    if(q_lora_rank == 0){
    assertTrue(hidden_dim == head_dim * num_heads,
               "hidden_dim != head_dim * num_heads");
    }
  };

  ModelConfig& operator=(const ModelConfig& rhs) = default;

  int hidden_dim;
  int head_dim;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int max_seq_len;
  int intermediate_dim;
  int expert_intermediate_dim;
  int activation_factor;
  int precision_byte;
  int num_routed_expert;
  int num_shared_expert;
  int expert_freq;
  int top_k;
  int ffn_way;
  int first_k_dense; // 0 for not use
  int q_lora_rank;     // for MLA
  int kv_lora_rank;    // for MLA
  int qk_nope_head_dim; // for MLA
  int qk_rope_head_dim; // for MLA  
  int n_vocab;
  
  bool compressed_kv;
  bool use_absorb;
  double skewness; // for Zipfian distribution
  std::string model_name;

  int ne_tp_dg;  // non-expert tensor parallelism degree
  int e_tp_dg;   // expert tensor parallelism degree
  std::string dataset;

  int input_len;
  int output_len;
};

static ModelConfig mixtral = ModelConfig(4096, 128, 32, 32, 8, 32768, 14336,
                                         14336, 1, 2, 8, 0, 1, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0, "mixtral");

static ModelConfig openMoE = ModelConfig(
    3072, 128, 32, 24, 24, 2048, 12288, 12288, 2, 2, 32, 0, 4, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0, "openMoE");

static ModelConfig llama7bMoE =
    ModelConfig(4096, 128, 32, 32, 32, 4096, 11008, 688, 1, 2, 16, 0, 1, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0,
                "llama7bMoE");

static ModelConfig grok1 = ModelConfig(6144, 128, 64, 48, 8, 8192, 32768, 32768,
                                       1, 2, 8, 0, 1, 2, 3, 0, 0, 0, 0, 0, 131072, false, false, 0.0, "grok1");

static ModelConfig glam = ModelConfig(4096, 128, 32, 32, 32, 8192, 16384, 16384,
                                      1, 2, 64, 0, 2, 2, 2, 0, 0, 0, 0, 0, 256000, false, false, 0.0, "glam");

static ModelConfig deepseekV3 =
    ModelConfig(7168, 128, 60, 128, 128, 131072, 18432, 2048, 1, 1, 256, 1, 1, 8,
                3, 3, 1536, 512, 128, 64, 129280, true, true, 0.0,"deepseekV3"); // n_layer = 60 (not consider MTP module)

static ModelConfig llama3_405B =
    ModelConfig(16384, 128, 126, 128, 8, 131072, 53248, 53248, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 128256, false, false, 0.0,
                "llama3_405B");

static ModelConfig llama4_scout = // 16 Expert 
    ModelConfig(5120, 128, 48, 40, 8, 10485760, 16384, 8192, 1, 2, 16, 1, 1, 1,
                3, 0, 0, 0, 0, 0, 202048, false, false, 0.0,"llama4_scout");
      
static ModelConfig llama4_maverick = // 128 Expert
                ModelConfig(5120, 128, 48, 40, 8, 1048576, 16384, 8192, 1, 2, 128, 1, 2, 1,
                            3, 0, 0, 0, 0, 0, 202048, false, false, 0.0,"llama4_maverick");

// Llama-3.1-8B: dense, GQA. hidden 4096 = 32 heads x 128 head_dim; 8 KV heads;
// 32 layers; SwiGLU FFN 14336; vocab 128256; 128K context. No experts/MLA.
static ModelConfig llama8B =
    ModelConfig(4096, 128, 32, 32, 8, 131072, 14336, 14336, 1, 1, 0, 0, 0, 0,
                3, 0, 0, 0, 0, 0, 128256, false, false, 0.0, "llama8B");
                
static ModelConfig llama8Bshort =
    ModelConfig(4096, 128, 1, 32, 8, 131072, 14336, 14336, 1, 1, 0, 0, 0, 0,
                3, 0, 0, 0, 0, 0, 128256, false, false, 0.0, "llama8Bshort");
// if model_config.q_lora_rank != 0 -> MLA로

}  // namespace llm_system