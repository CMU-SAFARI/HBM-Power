#include "module/layer.h"
#include "module/base.h"
#include "model/util.h"
#include "module/communication.h"
#include "module/route.h"

namespace llm_system {

// Attention //

Attention::Attention(std::string& prefix, std::string& name,
                     const ModelConfig& model_config, Scheduler::Ptr scheduler,
                     std::vector<int>& device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  auto attn_qkv_proj = ColumnParallelLinear::Create(
      module_map_name, "attn_qkv_proj", model_config.hidden_dim,
      model_config.head_dim *
          (model_config.num_heads + 2 * model_config.num_kv_heads),
      device_list, device);
  add_module(attn_qkv_proj);

  auto self_attention = SelfAttentionParallel::Create(
      module_map_name, "self_attention", model_config.head_dim,
      model_config.num_heads, model_config.num_kv_heads,
      model_config.input_len + model_config.output_len, scheduler->batch_size_per_dp, model_config.qk_rope_head_dim, 
      model_config.compressed_kv, device_list, device);
  add_module(self_attention);

  auto attn_o_proj =
      RowParallelLinear::Create(module_map_name, "attn_o_proj",
                                model_config.head_dim * model_config.num_heads,
                                model_config.hidden_dim, device_list, device);
  add_module(attn_o_proj);

  auto all_reduce =
      AllReduce::Create(module_map_name, "all_reduce", device_list, device);
  add_module(all_reduce);
}

Tensor::Ptr Attention::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attn_qkv_proj = get_module("attn_qkv_proj");
  Module::Ptr self_attention = get_module("self_attention");
  Module::Ptr attn_o_proj = get_module("attn_o_proj");
  Module::Ptr all_reduce = get_module("all_reduce");

  Tensor::Ptr qkv = (*attn_qkv_proj)(input, sequences_metadata);
  Tensor::Ptr attn_out = (*self_attention)(qkv, sequences_metadata);
  Tensor::Ptr o_proj = (*attn_o_proj)(attn_out, sequences_metadata);
  Tensor::Ptr result = (*all_reduce)(o_proj, sequences_metadata);

  return result;
}

// Multi Latent Attention //
MultiLatentAttention::MultiLatentAttention(std::string& prefix, std::string& name,
                                          const ModelConfig& model_config, Scheduler::Ptr scheduler,
                                          std::vector<int>& device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list) {

  use_absorb = model_config.use_absorb;
  num_heads = model_config.num_heads;
  parallel_num = device_list.size();

  auto attn_q_down_proj = Linear::Create(module_map_name, "attn_q_down_proj", model_config.hidden_dim,
      model_config.q_lora_rank, device_list, device);
  add_module(attn_q_down_proj);

  // or attn_kr_proj은 attn_kv_down_proj와 같이 연산될 수 있음, 현재는 DeepSeek-V3 demo inference code ver.과 다름//
  // auto attn_kv_down_proj = Linear::Create(module_map_name, "attn_kv_down_proj", model_config.hidden_dim,
  //   model_config.kv_lora_rank + model_config.qk_rope_head_dim, device_list, device);
  // add_module(attn_kv_down_proj);
  
  auto attn_kv_down_proj = Linear::Create(module_map_name, "attn_kv_down_proj", model_config.hidden_dim,
      model_config.kv_lora_rank, device_list, device);
  add_module(attn_kv_down_proj);

  auto attn_kr_proj = Linear::Create(module_map_name, "attn_kr_proj", model_config.hidden_dim,
      model_config.qk_rope_head_dim, device_list, device);
  add_module(attn_kr_proj);

  auto k_rope = RoPE::Create(module_map_name, "k_rope", model_config.num_heads,
    model_config.qk_rope_head_dim, device_list, device);
  add_module(k_rope);
  
  auto latent_q_layer_norm = LayerNorm::Create(module_map_name, "latent_q_layer_norm", model_config.q_lora_rank,
    device_list, device);
  add_module(latent_q_layer_norm);
  
  auto latent_kv_layer_norm = LayerNorm::Create(module_map_name, "latent_kv_layer_norm", model_config.kv_lora_rank,
    device_list, device);
  add_module(latent_kv_layer_norm);

  // for absorb MLA
  if(model_config.use_absorb){ 
    auto attn_q_up_proj = BatchedColumnParallelLinear::Create(
        module_map_name, "attn_q_up_proj", model_config.q_lora_rank,
        model_config.num_heads * model_config.qk_nope_head_dim, model_config.num_heads,
        true, false, device_list, device);
    add_module(attn_q_up_proj);

    auto attn_qr_up_proj = BatchedColumnParallelLinear::Create(
        module_map_name, "attn_qr_proj", model_config.q_lora_rank,
        model_config.num_heads * model_config.qk_rope_head_dim, model_config.num_heads,
        true, false, device_list, device);
    add_module(attn_qr_up_proj);

    auto q_rope = BatchedRoPE::Create(module_map_name, "q_rope", model_config.num_heads,
      model_config.qk_rope_head_dim, device_list, device);
    add_module(q_rope);

    auto attn_tr_k_up_proj = BatchedRowParallelLinear::Create(module_map_name, 
        "attn_tr_k_up_proj", // base: 512 x 16384, tr: 16384 x 512
        model_config.head_dim * model_config.num_heads,
        model_config.kv_lora_rank, model_config.num_heads, false, false, device_list, device);
    add_module(attn_tr_k_up_proj);

    auto attn_mla_absorbed = AbsorbMLAParallel::Create(module_map_name, "attn_mla_absorbed", model_config.head_dim,
      model_config.num_heads, model_config.num_kv_heads, model_config.input_len + model_config.output_len, 
      scheduler->batch_size_per_dp, model_config.qk_rope_head_dim, model_config.kv_lora_rank, model_config.compressed_kv,
      scheduler->system_config.use_flash_mla, device_list, device);
    add_module(attn_mla_absorbed);

    auto attn_v_up_proj = BatchedColumnParallelLinear::Create(
        module_map_name, "attn_v_up_proj", model_config.kv_lora_rank,
        model_config.num_heads * model_config.head_dim, model_config.num_heads,
        false, false, device_list, device);
    add_module(attn_v_up_proj);

    auto attn_o_up_proj = BatchedRowParallelLinear::Create(
        module_map_name, "attn_o_proj", model_config.num_heads * model_config.qk_nope_head_dim, 
        model_config.hidden_dim, model_config.num_heads, false, true, device_list, device);
    add_module(attn_o_up_proj);

    auto all_reduce = AllReduce::Create(module_map_name, "all_reduce", device_list, device);
    add_module(all_reduce);
  }
  else{
    // for baseline MLA
    auto attn_q_up_proj = ColumnParallelLinear::Create(
        module_map_name, "attn_q_up_proj", model_config.q_lora_rank,
        model_config.num_heads * model_config.qk_nope_head_dim,
        device_list, device);
    add_module(attn_q_up_proj);

    auto attn_qr_proj = ColumnParallelLinear::Create(
      module_map_name, "attn_qr_proj", model_config.q_lora_rank,
      model_config.num_heads * model_config.qk_rope_head_dim,
      device_list, device);
    add_module(attn_qr_proj);

    auto q_rope = RoPE::Create(module_map_name, "q_rope", model_config.num_heads,
      model_config.qk_rope_head_dim, device_list, device);
    add_module(q_rope);
    
    auto compressed_kv_restore = CompressedKVRestore::Create(
      module_map_name, "c_kv_restore", model_config.head_dim, model_config.num_heads, 
      model_config.input_len + model_config.output_len, scheduler->batch_size_per_dp,
      model_config.kv_lora_rank, model_config.qk_rope_head_dim, model_config.compressed_kv, device_list, device);
    add_module(compressed_kv_restore);
    
    auto attn_kv_up_proj = ColumnParallelLinear::Create(
        module_map_name, "attn_kv_up_proj", model_config.kv_lora_rank,
        model_config.num_heads * 2 * model_config.head_dim,
        device_list, device);
    add_module(attn_kv_up_proj);
  
    auto self_attention = MultiLatentAttentionParallel::Create(
        module_map_name, "multi_latent_attention", model_config.head_dim,
        model_config.num_heads, model_config.num_kv_heads, model_config.input_len + model_config.output_len, 
        scheduler->batch_size_per_dp, model_config.qk_rope_head_dim, model_config.compressed_kv, 
        scheduler->system_config.use_flash_mla, scheduler->system_config.use_flash_attention, device_list, device);
    add_module(self_attention);

    auto attn_o_proj =
    RowParallelLinear::Create(module_map_name, "attn_o_proj",
                model_config.head_dim * model_config.num_heads,
                model_config.hidden_dim, device_list, device);
    add_module(attn_o_proj);

    auto all_reduce =
    AllReduce::Create(module_map_name, "all_reduce", device_list, device);
    add_module(all_reduce);
  }
}


Tensor::Ptr MultiLatentAttention::forward(const Tensor::Ptr input,
              BatchedSequence::Ptr sequences_metadata) {
                
  Module::Ptr attn_q_down_proj = get_module("attn_q_down_proj");
  Module::Ptr attn_kv_down_proj = get_module("attn_kv_down_proj");
  Module::Ptr attn_kr_proj = get_module("attn_kr_proj");
  Module::Ptr k_rope = get_module("k_rope");

  Module::Ptr latent_q_layer_norm = get_module("latent_q_layer_norm");
  Module::Ptr latent_kv_layer_norm = get_module("latent_kv_layer_norm");

  if(use_absorb){
    Module::Ptr attn_q_up_proj = get_module("attn_q_up_proj");
    Module::Ptr attn_qr_proj = get_module("attn_qr_proj");
    Module::Ptr q_rope = get_module("q_rope");

    Module::Ptr attn_tr_k_up_proj = get_module("attn_tr_k_up_proj");
    Module::Ptr attn_mla_absorbed = get_module("attn_mla_absorbed");
    Module::Ptr attn_v_up_proj = get_module("attn_v_up_proj");
    Module::Ptr attn_o_proj = get_module("attn_o_proj");
    Module::Ptr all_reduce = get_module("all_reduce");

    Tensor::Ptr latent_q = (*attn_q_down_proj)(input, sequences_metadata);
    Tensor::Ptr latent_kv = (*attn_kv_down_proj)(input, sequences_metadata);
    Tensor::Ptr attn_kr_proj_out = (*attn_kr_proj)(input, sequences_metadata); //(S, 64)
    Tensor::Ptr k_rope_out = (*k_rope)(attn_kr_proj_out, sequences_metadata); // (S, 64)

    Tensor::Ptr q_ln_out = (*latent_q_layer_norm)(latent_q, sequences_metadata);
    Tensor::Ptr kv_ln_out = (*latent_kv_layer_norm)(latent_kv, sequences_metadata);
    
    TensorVec latent_query_vec;
    latent_query_vec.resize(0);
    for(int head_idx = 0; head_idx < (num_heads / parallel_num); head_idx ++){
      latent_query_vec.push_back(q_ln_out);
    }

    TensorVec query = (*attn_q_up_proj)(latent_query_vec, sequences_metadata); // (S, 16384) = (128, S, 128)
    TensorVec attn_qr_proj_out = (*attn_qr_proj)(latent_query_vec, sequences_metadata); // (S, 16384) = (128, S, 128)
    TensorVec q_rope_out = (*q_rope)(attn_qr_proj_out, sequences_metadata);

    TensorVec query_x_W_UK = (*attn_tr_k_up_proj)(query, sequences_metadata);

    TensorVec attn_mla_out = (*attn_mla_absorbed)(query_x_W_UK, sequences_metadata);

    TensorVec v_up_out = (*attn_v_up_proj)(attn_mla_out, sequences_metadata);
    TensorVec attn_out = (*attn_o_proj)(v_up_out, sequences_metadata);
    
    Tensor::Ptr output = attn_out[0]; // weighted_sum
    Tensor::Ptr result = (*all_reduce)(output, sequences_metadata);
    return result;
  }
  else{
    Module::Ptr c_kv_restore = get_module("c_kv_restore");
    
    Module::Ptr attn_q_up_proj = get_module("attn_q_up_proj");
    Module::Ptr attn_qr_proj = get_module("attn_qr_proj");
    Module::Ptr q_rope = get_module("q_rope");

    Module::Ptr attn_kv_up_proj = get_module("attn_kv_up_proj");

    Module::Ptr multi_latent_attention = get_module("multi_latent_attention");
    Module::Ptr attn_o_proj = get_module("attn_o_proj");
    Module::Ptr all_reduce = get_module("all_reduce");
    
    Tensor::Ptr latent_q = (*attn_q_down_proj)(input, sequences_metadata);
    Tensor::Ptr latent_kv = (*attn_kv_down_proj)(input, sequences_metadata);
    Tensor::Ptr attn_kr_proj_out = (*attn_kr_proj)(input, sequences_metadata); // TBD, implement RoPE (S, 64)
    Tensor::Ptr k_rope_out = (*k_rope)(attn_kr_proj_out, sequences_metadata); // (S, 64)


    Tensor::Ptr q_ln_out = (*latent_q_layer_norm)(latent_q, sequences_metadata);
    Tensor::Ptr kv_ln_out = (*latent_kv_layer_norm)(latent_kv, sequences_metadata);

    TensorVec kv_to_be_restored;
    kv_to_be_restored.resize(0);
    kv_to_be_restored.push_back(kv_ln_out);
    kv_to_be_restored.push_back(k_rope_out);
    
    TensorVec restored_kv = (*c_kv_restore)(kv_to_be_restored, sequences_metadata);
    Tensor::Ptr restored_latent_kv = restored_kv[0];
    Tensor::Ptr restored_k_rope = restored_kv[1];
    
    Tensor::Ptr query = (*attn_q_up_proj)(q_ln_out, sequences_metadata); // (S, 16384) = (128, S, 128)
    Tensor::Ptr q_rope_out = (*attn_qr_proj)(q_ln_out, sequences_metadata); // TBD, implement RoPE (S, 8192) = (128, S, 64)
    Tensor::Ptr key_value = (*attn_kv_up_proj)(restored_latent_kv, sequences_metadata);
    
    TensorVec attn_input;
    attn_input.resize(0);
    attn_input.push_back(query);
    attn_input.push_back(key_value);
    attn_input.push_back(q_rope_out);
    attn_input.push_back(k_rope_out); // 실제로는 RoPE의 output이 여기 input으로 들어가야함. 

    TensorVec attn_out = (*multi_latent_attention)(attn_input, sequences_metadata);
    Tensor::Ptr o_proj = (*attn_o_proj)(attn_out.at(0), sequences_metadata);

    Tensor::Ptr result = (*all_reduce)(o_proj, sequences_metadata);

    return result;
  }
}

// FeedForward //

FeedForward2Way::FeedForward2Way(std::string& prefix, std::string& name,
                                 const ModelConfig& model_config,
                                 Scheduler::Ptr scheduler,
                                 std::vector<int>& device_list,
                                 Device::Ptr device, bool perform_all_reduce,
                                 bool is_expert, bool use_dp)
    : Module(prefix, name, device, device_list),
      perform_all_reduce(perform_all_reduce) {
  int intermediate_dim = 0;
  if (is_expert) {
    intermediate_dim = model_config.expert_intermediate_dim;
  } else {
    intermediate_dim = model_config.intermediate_dim;
  }

  auto ffn_up_proj = ColumnParallelLinear::Create(
      module_map_name, "ffn_up_proj", model_config.hidden_dim,
      intermediate_dim * model_config.activation_factor, device_list, device);
  add_module(ffn_up_proj);

  auto act_fn = Activation::Create(module_map_name, "activation",
                                   model_config.activation_factor, device);
  add_module(act_fn);

  auto ffn_down_proj = RowParallelLinear::Create(
      module_map_name, "ffn_down_proj", intermediate_dim,
      model_config.hidden_dim, device_list, device);
  add_module(ffn_down_proj);

  if (perform_all_reduce) {
    auto all_reduce =
        AllReduce::Create(module_map_name, "all_reduce", device_list, device);
    add_module(all_reduce);
  }
}

Tensor::Ptr FeedForward2Way::forward(const Tensor::Ptr input,
                                     BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr ffn_up_proj = get_module("ffn_up_proj");
  Module::Ptr act_fn = get_module("activation");
  Module::Ptr ffn_down_proj = get_module("ffn_down_proj");

  Tensor::Ptr ffn_up = (*ffn_up_proj)(input, sequences_metadata);
  Tensor::Ptr act_out = (*act_fn)(ffn_up, sequences_metadata);
  Tensor::Ptr ffn_down = (*ffn_down_proj)(act_out, sequences_metadata);
  Tensor::Ptr result;

  if (perform_all_reduce) {
    Module::Ptr all_reduce = get_module("all_reduce");
    result = (*all_reduce)(ffn_down, sequences_metadata);
  } else {
    result = ffn_down;
  }

  return result;
}

// FeedForward3WAY //

FeedForward3Way::FeedForward3Way(std::string& prefix, std::string& name,
                                 const ModelConfig& model_config,
                                 Scheduler::Ptr scheduler,
                                 std::vector<int>& device_list,
                                 Device::Ptr device, bool perform_all_reduce,
                                 bool is_expert, bool use_dp)
    : Module(prefix, name, device, device_list),
      perform_all_reduce(perform_all_reduce) {
  int intermediate_dim = 0;
  if (is_expert) {
    intermediate_dim = model_config.expert_intermediate_dim;
  } else {
    intermediate_dim = model_config.intermediate_dim;
  }

  assertTrue(((perform_all_reduce == true) && (use_dp == true)) == false, "duplicated weight doesn't need all-reduce");
  if(!use_dp){
    auto w1 = ColumnParallelLinear::Create(
        module_map_name, "gate_proj", model_config.hidden_dim,
        intermediate_dim * model_config.activation_factor, device_list, device);
    add_module(w1);

    auto act_fn = Activation::Create(module_map_name, "activation",
                                    model_config.activation_factor, device);
    add_module(act_fn);

    auto w2 =
        RowParallelLinear::Create(module_map_name, "down_proj", intermediate_dim,
                                  model_config.hidden_dim, device_list, device);
    add_module(w2);

    auto w3 = ColumnParallelLinear::Create(module_map_name, "up_proj",
                                          model_config.hidden_dim,
                                          intermediate_dim, device_list, device);
    add_module(w3);

    if (perform_all_reduce) {
      auto all_reduce =
          AllReduce::Create(module_map_name, "all_reduce", device_list, device);
      add_module(all_reduce);
    }
  }
  else{
    auto w1 = Linear::Create(
        module_map_name, "gate_proj", model_config.hidden_dim,
        intermediate_dim * model_config.activation_factor, device_list, device);
    add_module(w1);

    auto act_fn = Activation::Create(module_map_name, "activation",
                                    model_config.activation_factor, device);
    add_module(act_fn);

    auto w2 =
        Linear::Create(module_map_name, "down_proj", intermediate_dim,
                                  model_config.hidden_dim, device_list, device);
    add_module(w2);

    auto w3 = Linear::Create(module_map_name, "up_proj",
                                          model_config.hidden_dim,
                                          intermediate_dim, device_list, device);
    add_module(w3);
  }
}

Tensor::Ptr FeedForward3Way::forward(const Tensor::Ptr input,
                                     BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr gate_proj = get_module("gate_proj");
  Module::Ptr up_proj = get_module("up_proj");
  Module::Ptr down_proj = get_module("down_proj");
  Module::Ptr act_fn = get_module("activation");

  Tensor::Ptr gate_out = (*gate_proj)(input, sequences_metadata);
  Tensor::Ptr act_out = (*act_fn)(gate_out, sequences_metadata);
  Tensor::Ptr up_proj_out = (*up_proj)(input, sequences_metadata);

  Tensor::Ptr ffn_out = (*down_proj)(up_proj_out, sequences_metadata);

  Tensor::Ptr result;
  if (perform_all_reduce) {
    Module::Ptr all_reduce = get_module("all_reduce");
    result = (*all_reduce)(ffn_out, sequences_metadata);
  } else {
    result = ffn_out;
  }

  return result;
}

}  // namespace llm_system