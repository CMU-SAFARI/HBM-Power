#include "hardware/device.h"

#include <random>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if(config.gpu_gen == "H100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
  }
  else if(config.gpu_gen == "A100"){
    dram_cfg_path = "./dram_config_HBM2_40GB.yaml";
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // B100, HBM3E 192GB, 8Gbps
    memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
    memory_config = hbm3e_192GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if(config.gpu_gen == "A100"){
    // A100 40GB, HBM2 1.2Gbps / 600MHz (timing preset HBM2E_1.2Gbps)
    memory_scale_factor = 1.6667; // tCK = 1/600MHz = 1.6667 ns (DRAM cycle -> ns)
    memory_config = hbm2_40GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  use_ramulator = config.use_ramulator;
  perform_execution = false;
}

void Device::set_dependency() { top_module_graph->set_dependency(); }

bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

void Device::restartGraph() { top_module_graph->restart_graph(); }

void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

void Device::reset_status() { status = StatusBoard(); }
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

void Device::setMemoryObject(Tensor::Ptr tensor) {
  mmap_controller->setMemoryObject(tensor);
}

void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());
  }
}

void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

void Device::run_ramulator(DRAMRequest_Ptr dram_request) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  dram_interface->HandleRequest(request, 0);
}

void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = (total_read % num_cube == 0) ? (total_read / num_cube) : ((total_read / num_cube) + 1);

  long rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 % num_channel == 0) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);
  long rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 % num_channel == 1) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);

  dram_interface->resetCounter();
  dram_interface->getExecStatus().act_count = (((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) < num_col) ? 1 : ((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) / num_col));
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel / 2;
  }
  else {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel;
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

};  // namespace llm_system