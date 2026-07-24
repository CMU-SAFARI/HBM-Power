
#include "lm_head.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// LmHead //

namespace llm_system {

LmHead::LmHead(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  std::vector<int> wgt_shape = {hidden_dimension, n_vocab};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr LmHead = Tensor::Create("lm_head_wgt", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(LmHead);
  add_tensor(output);
}

Tensor::Ptr LmHead::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int n = model_config.n_vocab;

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Hidden vector", shape);

  return output;
}

}  // namespace llm_system