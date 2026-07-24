#include "module/module_graph.h"

namespace llm_system {

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         Tensor::Ptr input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = false;
  stamped = false;
};

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         TensorVec input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input_vec(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = true;
  stamped = false;
};

bool ModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  if (module == nullptr || !module->execution()) {
    return true;
  } else if (check_ready() == true) {
    if (isTensorVec) {
      module->forward(input_vec, sequences_metadata);
    } else {
      module->forward(input, sequences_metadata);
    }
    // input->unset();
    return true;
  } else {
    return false;
  }
};

void ModuleGraph::set_dependency() { set_dependency_tensor(); }

bool ModuleGraph::checkListReady(TensorVec tensor_list) {
  for (Tensor::Ptr tensor : dependency_tensor_list) {
    if (tensor->ready == false) {
      return false;
    }
  }
  return true;
}

bool ModuleGraph::check_ready() {
  if (module->sync) {
    if (!checkListReady(dependency_tensor_list)) {
      return false;
    }
    // all operations are doned, we have to sync the devices
    sync_devices();
    return true;
  } else {
    if (input && input->ready) {
      return true;
    } else if (input_vec.size() != 0) {
      if (checkListReady(input_vec)) {
        return true;
      }
    }
  }
  return false;
}

void ModuleGraph::sync_devices() {
  // not yet synced
  if (input) {
    if (!input->timeboard_synced) {
      Device::Ptr device;
      time_ns time = 0;
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        device = tensor->get_device();
        time_ns device_time = device->get_time();
        time = std::max(time, device_time);
      }
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        tensor->timeboard_synced = true;
        device = tensor->get_device();
        device->set_time(time);
      }
      input->timeboard_synced = false;
    } else {
      input->timeboard_synced = false;
    }
  } else {
    fail("Module cannot be synced when it's inputs are TensorVector");
  }
}

void ModuleGraph::set_dependency_tensor() {
  // only when inputs are one Tensor pointer
  if (input && module && module->sync) {
    module->set_dependency_tensor(dependency_tensor_list, input);
  }
}

void ModuleGraph::print_graph() {
  if (module != nullptr) {
    for (int i = 0; i < module_level; i++) {
      std::cout << "\t";
    }
    std::cout << module->name << std::endl;
  }
}

TopModuleGraph::TopModuleGraph(StatusBoard& status)
    : status(status), module_graph(){};

void TopModuleGraph::push_module_graph(Module::Ptr module, Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::push_module_graph(Module::Ptr module, TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::set_dependency() {
  for (auto module : module_graph) {
    module->set_dependency();
  }
  restart_graph();
}

void TopModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  for (; current_module != module_graph.end(); current_module++) {
    set_stamp();
    // if execution is blocked because of sync
    if (!(*current_module)->run(sequences_metadata)) {
      break;
    }
  }
}

// deprecated
void TopModuleGraph::push_stamp() {
  fail("Cannot use push_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_pop()) {
    timeboard.push_timestamp(status, module_graph->get_name());
  }
}

// deprecated
void TopModuleGraph::pop_stamp() {
  fail("Cannot use pop_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (module_graph->is_pop()) {
    timeboard.pop_timestamp(status);
  }
}

void TopModuleGraph::set_stamp() {
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_stamped()) {
    if (module_graph->is_pop()) {
      set_pop_status();
      timeboard.pop_timestamp(status);
    } else {
      set_push_status();
      timeboard.push_timestamp(status, module_graph->get_name());
    }
    module_graph->set_stamped();
  }
}

void TopModuleGraph::set_push_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;

    status.device_time = std::max(status.device_time,
                                  std::max(status.low_time, status.high_time));
    status.low_time = status.device_time;
    status.high_time = status.device_time;
    status.parallel_execution = false;
    //

  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
      if (status.tensor->isPerformHigh()) {
        status.device_time = status.high_time;
      } else {
        status.device_time = status.low_time;
      }
    } else {
      status.device_time = std::max(
          status.device_time, std::max(status.low_time, status.high_time));
      status.low_time = status.device_time;
      status.high_time = status.device_time;
      status.parallel_execution = false;
    }
  }
}

void TopModuleGraph::set_pop_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;
  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
    } else {
      status.parallel_execution = false;
    }
  }

  ExecStatus exec_status = device->getExecStatus();

  if (status.parallel_execution) {
    if (exec_status.processor_type == ProcessorType::LOGIC ||
        exec_status.processor_type == ProcessorType::PIM) {
      status.low_time += exec_status.total_duration;
      status.device_time = status.low_time;
    } else if (exec_status.processor_type == ProcessorType::GPU) {
      status.high_time += exec_status.total_duration;
      status.device_time = status.high_time;
    }
  } else {
    status.device_time += exec_status.total_duration;
    // status.high_time = status.device_time;
    // status.low_time = status.device_time;
  }
  
  if (exec_status.processor_type == ProcessorType::PIM ||
      exec_status.processor_type == ProcessorType::LOGIC ||
      exec_status.processor_type == ProcessorType::GPU) {
    int processor_type = (int)exec_status.processor_type;
    status.act_energy +=
        exec_status.act_count * dram_powers[processor_type].kACT_energy_j_;
    status.read_energy +=
        exec_status.read_count * dram_powers[processor_type].kREAD_energy_j_;
    status.write_energy +=
        exec_status.write_count * dram_powers[processor_type].kWRITE_energy_j_;

    status.all_act_energy += exec_status.all_act_count *
                             dram_powers[processor_type].kALL_ACT_energy_j_;
    status.all_read_energy += exec_status.all_read_count *
                              dram_powers[processor_type].kALL_READ_energy_j_;
    status.all_write_energy += exec_status.all_write_count *
                               dram_powers[processor_type].kALL_WRITE_energy_j_;

    status.mac_energy +=
        exec_status.flops * dram_powers[processor_type].kMAC_energy_j_;
    ;  // 2flops per operation, energy per operation, pJ to nJ
  }

  // if (!exec_status.parallel_execution) {
  //   // status.device_time = std::max(status.device_time,
  //   //                               std::max(status.low_time,
  //   //                               status.high_time));
  //   // status.low_time = status.device_time;
  //   // status.high_time = status.device_time;
  //   status.parallel_execution = false;
  // } else {
  //   status.parallel_execution = true;
  // }

  // if (exec_status.processor_type == ProcessorType::LOGIC ||
  //     exec_status.processor_type == ProcessorType::PIM) {
  //   status.low_time += exec_status.total_duration;
  //   status.device_time = status.low_time;
  // } else if (exec_status.processor_type == ProcessorType::GPU) {
  //   status.high_time += exec_status.total_duration;
  //   status.device_time = status.high_time;
  // } else {
  //   status.low_time = status.device_time;
  //   status.high_time = status.device_time;
  // }

  status.compute_util = exec_status.compute_util;
  status.memory_util = exec_status.memory_util;
  status.processor_type = exec_status.processor_type;

  status.flops += exec_status.flops;
  status.memory_size += exec_status.memory_size;

  status.opb = exec_status.opb;
}

void TopModuleGraph::print_graph() {
  std::cout << "Print graph" << std::endl;
  for (auto module_graph_ : module_graph) {
    module_graph_->print_graph();
  }
}

void TopModuleGraph::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  if(dram_powers.size() == 0){
    for (int i = 0; i < (int)ProcessorType::MAX; i++) {
      DramEnergy temp;
      dram_powers.push_back(temp);
    }
  }
  dram_powers[ProcessorType] = dramEnergy;
}

std::vector<energy_nJ> TopModuleGraph::getDeviceEnergy(){
  std::vector<energy_nJ> device_energy {status.act_energy, status.read_energy, status.write_energy, 
                            status.all_act_energy, status.all_read_energy, status.all_write_energy,
                            status.mac_energy, status.act_energy + status.read_energy + status.write_energy + 
                            status.all_act_energy + status.all_read_energy + status.all_write_energy + status.mac_energy}; 
  return device_energy;
}

void TopModuleGraph::restart_graph() {
  current_module = module_graph.begin();
  assertTrue(current_module != module_graph.end(),
             "No module in TopModuleGraph");
  (*current_module)->set_ready();

  for (auto module_graph_ : module_graph) {
    module_graph_->unset_tensor();
  }
};

}  // namespace llm_system