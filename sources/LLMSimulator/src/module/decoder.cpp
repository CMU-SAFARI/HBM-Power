#include "module/decoder.h"

#include "model/util.h"
#include "module/expert.h"
#include "module/layer.h"
#include "module/communication.h"

namespace llm_system {

// Decoder //

Decoder::Decoder(std::string& prefix, std::string& name,
                 const ModelConfig& model_config, Scheduler::Ptr scheduler,
                 std::vector<int>& device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();

  int ne_tp_dg = model_config.ne_tp_dg;
  int ne_tp_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;
  std::vector<int> non_moe_device_list;
  set_device_list(non_moe_device_list, ne_tp_offset, ne_tp_dg);

  assertTrue(parallel_num >= ne_tp_dg,
             "None expert tensor parallel degree is not supported");
  
  // Input & Post-Attn LayerNorm is Standard now //
  auto input_layer_norm = LayerNorm::Create(module_map_name, "input_layer_norm", model_config.hidden_dim,
                non_moe_device_list, device);
  add_module(input_layer_norm);

  if(model_config.qk_rope_head_dim == 0){
    auto attention = Attention::Create(module_map_name, "attention", model_config,
                                      scheduler, non_moe_device_list, device);
    add_module(attention);
  }
  else{ // MLA
    auto attention = MultiLatentAttention::Create(module_map_name, "attention", model_config,
      scheduler, non_moe_device_list, device);
    add_module(attention);
  }

  auto residual_1 = Residual::Create(module_map_name, "residual_1", model_config.hidden_dim,
     non_moe_device_list, device);
  add_module(residual_1);

  auto post_attn_layer_norm = LayerNorm::Create(module_map_name, "post_attn_layer_norm", model_config.hidden_dim,
      non_moe_device_list, device);
  add_module(post_attn_layer_norm);

  if (model_config.ffn_way == 2) {
    auto feedforward =
        FeedForward2Way::Create(module_map_name, "feedforward", model_config,
                                scheduler, non_moe_device_list, device, false);
    add_module(feedforward);
  } else if (model_config.ffn_way == 3) {
    auto feedforward =
        FeedForward3Way::Create(module_map_name, "feedforward", model_config,
                                scheduler, non_moe_device_list, device, false);
    add_module(feedforward);
  }

  auto all_reduce = AllReduce::Create(module_map_name, "all_reduce", non_moe_device_list, device);
  add_module(all_reduce);

  auto residual_2 = Residual::Create(module_map_name, "residual_2", model_config.hidden_dim,
      non_moe_device_list, device);
  add_module(residual_2);
}

Tensor::Ptr Decoder::forward(const Tensor::Ptr input,
                             BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr input_layer_norm = get_module("input_layer_norm");
  Module::Ptr attention = get_module("attention");
  Module::Ptr residual_1 = get_module("residual_1");
  Module::Ptr post_attn_layer_norm = get_module("post_attn_layer_norm");
  Module::Ptr feedforward = get_module("feedforward");
  Module::Ptr all_reduce = get_module("all_reduce");
  Module::Ptr residual_2 = get_module("residual_2");

  Tensor::Ptr input_ln_out = (*input_layer_norm)(input, sequences_metadata);
  Tensor::Ptr attention_out = (*attention)(input_ln_out, sequences_metadata);
  Tensor::Ptr res_1_out = (*residual_1)(attention_out, sequences_metadata);
  Tensor::Ptr post_attn_ln_out = (*post_attn_layer_norm)(attention_out, sequences_metadata);
  Tensor::Ptr ffn_out = (*feedforward)(post_attn_ln_out, sequences_metadata);
  Tensor::Ptr result = (*all_reduce)(ffn_out, sequences_metadata);
  Tensor::Ptr res_2_out = (*residual_1)(result, sequences_metadata);


  return res_2_out;
}

MoEDecoder::MoEDecoder(std::string& prefix, std::string& name,
                       const ModelConfig& model_config,
                       Scheduler::Ptr scheduler, std::vector<int>& device_list,
                       Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();

  int ne_tp_dg = model_config.ne_tp_dg;
  int ne_tp_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;

  std::vector<int> non_moe_device_list;
  set_device_list(non_moe_device_list, ne_tp_offset, ne_tp_dg);
  
  assertTrue(parallel_num >= ne_tp_dg,
             "None expert tensor parallel degree is not supported");

  // Input & Post-Attn LayerNorm is Standard now //
  auto input_layer_norm = LayerNorm::Create(module_map_name, "input_layer_norm", model_config.hidden_dim,
    non_moe_device_list, device);
  add_module(input_layer_norm);
  
  if(model_config.qk_rope_head_dim == 0){
    auto attention = Attention::Create(module_map_name, "attention", model_config,
                                      scheduler, non_moe_device_list, device);
    add_module(attention);
  }
  else{ // MLA
    auto attention = MultiLatentAttention::Create(module_map_name, "attention", model_config,
      scheduler, non_moe_device_list, device);
    add_module(attention);
  }

  auto residual_1 = LayerNorm::Create(module_map_name, "residual_1", model_config.hidden_dim,
      non_moe_device_list, device);
  add_module(residual_1);

  auto post_attn_layer_norm = LayerNorm::Create(module_map_name, "post_attn_layer_norm", model_config.hidden_dim,
    non_moe_device_list, device);
  add_module(post_attn_layer_norm);

  set_device_list(device_list, 0, device_list.size());
  auto expert_ffn =
      ExpertFFN::Create(module_map_name, "expertFFN", model_config, scheduler,
                        device_list, device);
  add_module(expert_ffn);

  auto residual_2 = LayerNorm::Create(module_map_name, "residual_2", model_config.hidden_dim,
      non_moe_device_list, device);
  add_module(residual_2);
}

Tensor::Ptr MoEDecoder::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr input_layer_norm = get_module("input_layer_norm");
  Module::Ptr attention = get_module("attention");
  Module::Ptr residual_1 = get_module("residual_1");
  Module::Ptr post_attn_layer_norm = get_module("post_attn_layer_norm");
  Module::Ptr expert_ffn = get_module("expertFFN");
  Module::Ptr residual_2 = get_module("residual_2");
  

  Tensor::Ptr input_ln_out = (*input_layer_norm)(input, sequences_metadata);
  Tensor::Ptr attention_out = (*attention)(input_ln_out, sequences_metadata);
  Tensor::Ptr res_1_out = (*residual_1)(attention_out, sequences_metadata);
  Tensor::Ptr post_attn_ln_out = (*post_attn_layer_norm)(attention_out, sequences_metadata);
  Tensor::Ptr result = (*expert_ffn)(post_attn_ln_out, sequences_metadata);
  Tensor::Ptr res_2_out = (*residual_2)(result, sequences_metadata);

  return res_2_out;
  // return result;
}

}  // namespace llm_system