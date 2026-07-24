#include "hardware/node.h"

namespace llm_system {

Node::Node(SystemConfig config, int node_rank, Cluster_ptr cluster)
    : config(config), node_rank(node_rank), cluster(cluster) {
  node_ict_latency = config.device_ict_latency;
  node_ict_bandwidth = config.device_ict_bandwidth;

  CreateDevice(config, cluster);
}

Device::Ptr Node::get_device(int device_total_rank) {
  int local_rank = device_total_rank % config.num_device;
  return device.at(local_rank);
}

void Node::set_dependency() {
  for (Device::Ptr _device : device) {
    _device->set_dependency();
  }
}

bool Node::check_module_graph_remain() {
  for (Device::Ptr _device : device) {
    if (_device->check_module_graph_remain()) {
      return true;
    }
  }
  return false;
}

void Node::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  for (Device::Ptr _device : device) {
      _device->run(sequences_metadata_list);
  }
}

void Node::setPerformExecution(bool perform) {
  for (Device::Ptr _device : device) {
    _device->setPerformExecution(perform);
  }
};

void Node::CreateDevice(SystemConfig config, Cluster_ptr cluster) {
  int device_rank_offset = node_rank * config.num_device;
  for (int device_rank = 0; device_rank < config.num_device; device_rank++) {
    device.push_back(
        Device::Create(config, device_rank + device_rank_offset, cluster));
  }
}

};  // namespace llm_system