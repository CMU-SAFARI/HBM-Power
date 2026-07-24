#pragma once
#include <map>
#include <string>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "module/base.h"
#include "module/module.h"
#include "module/status.h"
#include "module/timeboard.h"
#include "scheduler/scheduler.h"
#include "dram/power.h"

namespace llm_system {

// block of modules which can run simultaneosuly without communication and
// synchronizations

class TopModuleGraph;

class ModuleGraph : std::enable_shared_from_this<ModuleGraph> {
 public:
  friend class TopModuleGraph;
  using Ptr = std::shared_ptr<ModuleGraph>;

  [[nodiscard]] static Ptr Create(StatusBoard& status,
                                  Module::Ptr module = nullptr,
                                  Tensor::Ptr input = nullptr,
                                  int module_level = 0,
                                  bool module_pop = false) {
    return Ptr(
        new ModuleGraph(module, status, input, module_level, module_pop));
  };

  [[nodiscard]] static Ptr Create(StatusBoard& status,
                                  Module::Ptr module = nullptr,
                                  TensorVec input = {}, int module_level = 0,
                                  bool module_pop = false) {
    return Ptr(
        new ModuleGraph(module, status, input, module_level, module_pop));
  };

  bool run(BatchedSequence::Ptr sequences_metadata);

  void print_graph();

  std::string getname() { return module->name; };

  bool check_ready();
  void set_ready() { input->set(); };

  void set_dependency();

  void unset_tensor() {
    if (module) {
      module->unset_tensor();
    }
    unset_stamped();
  }

  bool is_stamped() { return stamped; }
  void unset_stamped() { stamped = false; }
  void set_stamped() { stamped = true; }

  std::string get_name() { return module->name; }

  void set_dependency_tensor();

  bool is_pop() { return module_pop; }

  void sync_devices();

  bool checkListReady(TensorVec tensor_list);

  std::vector<Tensor::Ptr> dependency_tensor_list;

 private:
  ModuleGraph(Module::Ptr module, StatusBoard& status, Tensor::Ptr input,
              int module_level, bool module_pop);
  ModuleGraph(Module::Ptr module, StatusBoard& status, TensorVec input,
              int module_level, bool module_pop);

  Tensor::Ptr input;
  TensorVec input_vec;
  Tensor::Ptr output;
  TensorVec output_vec;

  bool isTensorVec;

  int module_level;

  bool module_pop;
  bool stamped;

  StatusBoard& status;

  Module::Ptr module;
};

class TopModuleGraph {
 public:
  using Ptr = std::shared_ptr<TopModuleGraph>;

  [[nodiscard]] static Ptr Create(StatusBoard& status) {
    return Ptr(new TopModuleGraph(status));
  };

  std::vector<ModuleGraph::Ptr> module_graph;

  void connectDevice(Device::Ptr device_) { device = device_; }

  void push_module_graph(Module::Ptr module, Tensor::Ptr input);
  void pop_module_graph(Tensor::Ptr input);

  void push_module_graph(Module::Ptr module, TensorVec input);
  void pop_module_graph(TensorVec input);

  void run(BatchedSequence::Ptr sequences_metadata);
  void restart_graph();

  void set_dependency();
  void print_graph();

  void reset_timeboard() {
    timeboard.reset_timeboard();
  }

  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);
  std::vector<energy_nJ> getDeviceEnergy();

  bool check_module_graph_remain() {
    if (current_module == module_graph.end()) {
      return false;
    } else {
      return true;
    }
  }

  void print_timeboard() { timeboard.print(); }

  void exportGantt(std::string filepath, int device_id) {
    timeboard.exportGantt(filepath, device_id);
  };

  Device::Ptr device;

  TimeBoard timeboard;

 private:
  TopModuleGraph(StatusBoard& status);
  void set_push_status();
  void set_pop_status();

  void set_stamp();
  void push_stamp();
  void pop_stamp();

  StatusBoard& status;

  std::vector<DramEnergy> dram_powers;

  int current_module_level = 0;
  std::vector<ModuleGraph::Ptr>::iterator current_module;
};

}  // namespace llm_system