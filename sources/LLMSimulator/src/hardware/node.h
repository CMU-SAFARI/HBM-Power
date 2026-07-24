#pragma once
#include <memory>
#include <vector>

#include "common/type.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Cluster;

class Node : public std::enable_shared_from_this<Node> {
 public:
  using Ptr = std::shared_ptr<Node>;
  using Cluster_ptr = std::shared_ptr<Cluster>;

  [[nodiscard]] static Ptr Create(SystemConfig config, int node_rank,
                                  Cluster_ptr cluster) {
    return Ptr(new Node(config, node_rank, cluster));
  }

  hw_metric node_ict_latency;
  hw_metric node_ict_bandwidth;
  int node_rank;

  std::vector<Device::Ptr> device;

  SystemConfig config;

  void InitNode(int num_device);

  Node(Node &&) = default;
  Node &operator=(Node &&) = default;

  void set_dependency();
  bool check_module_graph_remain();
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);

  void setPerformExecution(bool perform);

  Device::Ptr get_device(int device_total_rank);

 private:
  Cluster_ptr cluster;
  Node(SystemConfig config, int node_rank, Cluster_ptr cluster);
  Node() = default;
  void CreateDevice(SystemConfig config, Cluster_ptr cluster);
};

}  // namespace llm_system