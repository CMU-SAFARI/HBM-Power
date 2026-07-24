#include "hardware/cluster.h"
#include "module/layer.h"
#include "scheduler/scheduler.h"
#include "scheduler/sequence.h"

using namespace llm_system;

int main() {
  BatchedSequence a;
  a.add_dummy_sequence(64, 256, 512, 2, "./expert_idx_file_GLaM.txt");

  ModelConfig model_config(4096, 128, 1, 32, 32); // hidden_dim, head_dim, num_layer, num_heads, num_kv_heads

  Scheduler::Ptr scheduler = Scheduler::Create(64, 8192, 2048); // batch, num_max_batched_token, max_process_token, num_expert
  scheduler->initializeDummyInput(1, 64, 2048, 4096, 2, 8,
                                  "./expert_idx_file_GLaM.txt", scheduler);  // batch,

  scheduler->step();  // sum
  scheduler->step();  // gen

  std::vector<int> device_list = {1};
  // std::cout << device_list.size() << std::endl;

  SystemConfig config;
  Cluster::Ptr cluster;
  Device::Ptr device = Device::Create(config, 0, cluster);

  Attention::Ptr attn = Attention::Create("", "Attention_Layer", model_config,
                                          scheduler, device_list, device);

  long size = attn->size();

  Tensor::Ptr input_tensor = Tensor::Create("test", {64, 4096}, "act", device->model_config.precision_byte);

  (*attn)(input_tensor, scheduler->running_queue[0]);

  FeedForward3way::Ptr ffn = FeedForward::Create(
      "", "FeedForward_Layer", model_config, scheduler, device_list, device);

  input_tensor = Tensor::Create("test", {64, 4096}, "act");

  (*ffn)(input_tensor, scheduler->running_queue[0]);

  std::cout << "Total size: " << size << std::endl;
  return 0;
}
