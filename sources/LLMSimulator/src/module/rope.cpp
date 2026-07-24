#include "module/rope.h"
#include "scheduler/scheduler.h"
#include "common/assert.h"

namespace llm_system {

RoPE::RoPE(std::string& prefix, std::string& name, int num_heads,
                     int qk_rope_head_dim, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
    qk_rope_head_dim(qk_rope_head_dim){
    std::vector<int> shape = {1, 1};

    Tensor::Ptr precomputed_sin_cos = Tensor::Create("sin_cos", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr rope_out = Tensor::Create("rope_out", shape, "act", device, device->model_config.precision_byte);
    add_tensor(precomputed_sin_cos);
    add_tensor(rope_out);
}

Tensor::Ptr RoPE::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  // only need max length seq's sin & cos 
  // if a seq is shorter than this, it can be covered by longer seq
  int max_len = 0; 
  auto seq_list = sequences_metadata->get_seq();
  for (auto seq : seq_list){
    max_len = std::max(seq->current_len + seq->num_process_token, 0);
  }
  std::vector<int> sin_cos_shape = {2 * max_len, qk_rope_head_dim};

  Tensor::Ptr precomputed_cos_sin = get_activation("sin_cos", sin_cos_shape);
  Tensor::Ptr output = get_activation("rope_out", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  double flops, memory_size;
  flops = m * k * 6; // rope = 6 ops (q,k elemwise mul (1), mul with rotate_half (1), add (1))
  memory_size = (2.0 * m * k + 1.0 * sin_cos_shape[0] * sin_cos_shape[1]) * input->precision_byte;

  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = std::max(compute_duration, memory_duration);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  return output;
}

BatchedRoPE::BatchedRoPE(std::string& prefix, std::string& name, int num_heads,
    int qk_rope_head_dim, std::vector<int> device_list, Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  qk_rope_head_dim(qk_rope_head_dim){
  std::vector<int> shape = {1, 1};

  Tensor::Ptr precomputed_sin_cos = Tensor::Create("sin_cos", shape, "act", device, device->model_config.precision_byte);
  Tensor::Ptr rope_out = Tensor::Create("rope_out", shape, "act", device, device->model_config.precision_byte);
  add_tensor(precomputed_sin_cos);
  add_tensor(rope_out);
}

TensorVec BatchedRoPE::forward(const TensorVec input,
            BatchedSequence::Ptr sequences_metadata) {
  
  int m = input.size();
  int k = input[0]->shape[0];
  int n = input[0]->shape[1];

  // only need max length seq's sin & cos 
  // if a seq is shorter than this, it can be covered by longer seq
  int max_len = 0; 
  auto seq_list = sequences_metadata->get_seq();
  for (auto seq : seq_list){
  max_len = std::max(seq->current_len + seq->num_process_token, 0);
  }
  std::vector<int> sin_cos_shape = {2 * max_len, qk_rope_head_dim};

  Tensor::Ptr precomputed_cos_sin = get_activation("sin_cos", sin_cos_shape);
  Tensor::Ptr output = get_activation("rope_out", input[0]->shape);

  long size = input[0]->getSize();
  if (size == 0) {
  return input;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  double flops, memory_size;
  flops = m * k * n * 6; // rope = 6 ops (q,k elemwise mul (1), mul with rotate_half (1), add (1))
  memory_size = (2.0 * m * k * n + 1.0 * sin_cos_shape[0] * sin_cos_shape[1]) * input[0]->precision_byte;

  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = std::max(compute_duration, memory_duration);

  if (input[0]->parallel_execution && !device->config.communication_hiding) {
    if (input[0]->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
    }
    device->status.device_time += total_time;
  return input;
}

}  // namespace llm_system