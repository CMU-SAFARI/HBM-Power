
#include "embedding.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// Embedding //

namespace llm_system {

Embedding::Embedding(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  std::vector<int> wgt_shape = {n_vocab, hidden_dimension};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr embedding = Tensor::Create("Embedding", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(embedding);
  add_tensor(output);
}

Tensor::Ptr Embedding::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int n = model_config.hidden_dim;

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Hidden vector", shape);

  return output;
}

}  // namespace llm_system