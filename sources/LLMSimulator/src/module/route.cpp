
#include "route.h"

#include <algorithm>
#include <map>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "hardware/hardware_config.h"
#include "module/module_graph.h"

// Route //
// take tensor as a input, and make mutliple tensors as output

namespace llm_system {

bool cmp(std::pair<int, int>& a, std::pair<int, int>& b) {
  return a.second > b.second;
}

bool cmp_vec(std::pair<std::vector<int>, std::vector<int>>& a, std::pair<std::vector<int>, std::vector<int>>& b) {
  int a_sum = 0;
  for (int i : a.second){
    a_sum += i;
  }
  int b_sum =0;
  for (int i : b.second){
    b_sum += i;
  }

  return a_sum > b_sum;
}

std::vector<int> getIdxHighOptimal(std::vector<int> list, SystemConfig config) {
  // expert_idx, num_token_per_expert
  double high_time = 0;
  double low_time = 0;
  double high_opb =
      config.compute_peak_flops / (config.memory_bandwidth * 0.75);

  std::vector<std::pair<int, int>> expert_list;
  int i = 0;
  for (auto element : list) {
    expert_list.push_back(std::pair<int, int>(i++, element));
    low_time += std::max(1.0, 1.0 * element / config.logic_op_b * 0.75) /
                config.logic_x;
  }

  sort(expert_list.begin(), expert_list.end(), cmp);
  std::vector<int> high_list;
  high_list.resize(0);
  double max = std::max(high_time, low_time);

  for (auto expert : expert_list) {
    int num_token = expert.second;
    high_time += std::max(1.0, 1.0 * num_token / high_opb);
    low_time -= std::max(1.0, 1.0 * num_token / config.logic_op_b * 0.75) /
                config.logic_x;
    double temp = std::max(high_time, low_time);
    if (max > temp) {
      high_list.push_back(expert.first);
      max = temp;
    } else {
      break;
    }
  }
  return high_list;
}

std::vector<int> getIdxHigh(std::vector<int> list, SystemConfig config) {
  // expert_idx, num_token_per_expert

  double high_time = 0;
  double low_time = 0;
  double high_opb =
      config.compute_peak_flops / (config.memory_bandwidth * 0.75);

  std::vector<std::pair<std::vector<int>, std::vector<int>>> expert_list;

  // expert group size
  if (list.size() <= 4) {
    return getIdxHighOptimal(list, config);
  }

  assertTrue(list.size() % 4 == 0, "Expert group size is not mached");

  int group_size = list.size() / 4;

    std::vector<int> expert_id_vec;
    std::vector<int> element_num_vec;
  for (int i = 0; i < list.size(); i+= group_size){
    expert_id_vec.resize(0);
    element_num_vec.resize(0);
    int temp = 0;
    for (int id = 0; id < group_size; id++){
      int element = list.at(i + id);
      expert_id_vec.push_back(i + id);
      element_num_vec.push_back(element);
      low_time += std::max(1.0, 1.0 * element / config.logic_op_b * 0.75) /
                  config.logic_x;
    }
    expert_list.push_back(std::pair<std::vector<int>, std::vector<int>>(expert_id_vec, element_num_vec));
  }


  sort(expert_list.begin(), expert_list.end(), cmp_vec);
  std::vector<int> high_list;
  high_list.resize(0);
  double max = std::max(high_time, low_time);

  for (auto expert : expert_list) {
    for (int num_token: expert.second){
    high_time += std::max(1.0, 1.0 * num_token / high_opb);
    low_time -= std::max(1.0, 1.0 * num_token / config.logic_op_b * 0.75) /
                config.logic_x;
    }
    double temp = std::max(high_time, low_time);
    if (max > temp) {
      for (int id : expert.first)
      high_list.push_back(id);
      max = temp;
    } else {
      break;
    }
  }
  return high_list;
}

GateUpdate::GateUpdate(std::string& prefix, std::string& name,
                       std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  // initializing expert_input //
  std::vector<int> shape = {1, 1};
  Tensor::Ptr gate_update_input =
      Tensor::Create("gate_update_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(gate_update_input);
}

Tensor::Ptr GateUpdate::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  int num_expert = sequences_metadata->num_expert;

  Tensor::Ptr output = get_activation("gate_update_output", input->shape);
  aggregate_expert(sequences_metadata);

  return output;
}

void GateUpdate::aggregate_expert(BatchedSequence::Ptr sequences_metadata) {
  // not yet aggregated
  int num_expert = device->model_config.num_routed_expert;
  int top_k = device->model_config.top_k;

  if (device->device_total_rank == 0) {
    // update each sequences_metadata
    Scheduler::Ptr scheduler = sequences_metadata->scheduler;
    if (scheduler == nullptr) {
      return;
    }
    int num_batch = scheduler->running_queue.size();
    for (int i = 0; i < num_batch; i++) {
      scheduler->running_queue[i]->update_expert(num_expert, top_k, true);
    }

    // aggregate whole sequences_metadata's expert
    std::vector<int> temp(num_expert);
    for (int i = 0; i < num_batch; i++) {
      for (int j = 0; j < num_expert; j++) {
        scheduler->running_queue[i]->local_num_token_in_expert[j] = scheduler->running_queue[i]->num_token_in_expert[j];
        temp[j] += scheduler->running_queue[i]->num_token_in_expert[j];
      }
    }

    // update whole sequences_metadata's expert
    int test_token = 0;
    for (int i = 0; i < num_batch; i++) {
      for (int j = 0; j < num_expert; j++) {
        scheduler->running_queue[i]->num_token_in_expert[j] = temp[j];
        test_token += temp[j];
      }
    }

    assertTrue(test_token == scheduler->getNumProcessToken() *
                                 device->model_config.top_k * num_batch,
               "Number of tokens in experts is not macthed, expected " +
                   std::to_string(scheduler->getNumProcessToken() *
                                  device->model_config.top_k) +
                   ", but" + std::to_string(test_token));
  }
}

Route::Route(std::string& prefix, std::string& name, int num_expert_per_device,
             int expert_offset, std::vector<int> device_list,
             Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      expert_offset(expert_offset),
      num_expert_per_device(num_expert_per_device) {
  
 int num_expert = device->model_config.num_routed_expert;
 int hidden_dim = device->model_config.hidden_dim;
 std::vector<int> shape = {hidden_dim, num_expert};
  Tensor::Ptr expert_input = Tensor::Create(
    "route_weight", shape, "weight", device, device->model_config.precision_byte);
  add_tensor(expert_input);

  // initializing expert_input //
  for (int expert_idx = 0; expert_idx < num_expert; expert_idx++) {
    std::vector<int> shape = {1, 1};
    Tensor::Ptr expert_input = Tensor::Create(
        "expert_input_" + std::to_string(expert_idx), shape, "act", device, device->model_config.precision_byte);
    add_tensor(expert_input);
  }
}

TensorVec Route::forward(const TensorVec input,
                         BatchedSequence::Ptr sequences_metadata) {
  TensorVec output_vec;
  
  std::vector<int> expert_token_list;
  int num_expert = device->model_config.num_routed_expert;
  for (int expert_idx = expert_offset;
       expert_idx < expert_offset + num_expert_per_device; expert_idx++) {
    expert_token_list.push_back(
        sequences_metadata->num_token_in_expert[expert_idx]);
  }

  for (int expert_idx = 0; expert_idx < num_expert; expert_idx++) {
    std::vector<int> shape = {
        sequences_metadata->num_token_in_expert[expert_idx],
        input[0]->shape[1]};
    Tensor::Ptr output =
        get_activation("expert_input_" + std::to_string(expert_idx), shape);
    output_vec.push_back(output);

    if (device->config.hetero_subbatch) {
      output->setPerformLow();
    } else if (device->config.parallel_execution) {
      std::vector<int> highExpert = getIdxHigh(expert_token_list, device->config);
      if (find(highExpert.begin(), highExpert.end(),
               expert_idx - expert_offset) != highExpert.end()) {
        output->setPerformHigh();
      } else {
        output->setPerformLow();
      }
    } else if(device->config.use_low_unit_moe_only){
      output->setPerformLow();
    } else{
      output->perform_with_optimal = true;
    }
  }

  return output_vec;
}

}  // namespace llm_system