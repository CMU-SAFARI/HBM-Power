#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

int main() {
  int num_device = 4;
  SystemConfig system_config;
  system_config.num_node = 1;
  system_config.num_device = num_device;
  system_config.processor_type = {ProcessorType::GPU};
  system_config.parallel_execution = false;
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;
  system_config.use_ramulator = false;
  system_config.communication_hiding = false;

  ModelConfig model_config;
  model_config = openMoE;
  model_config.ne_tp_dg = num_device;
  model_config.e_tp_dg = num_device;

  model_config.dataset = "GSM";
  long max_batch_size = 128;
  long total_batch_size = 64;

  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, "", max_batch_size, 4096);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  cluster->checkMemorySize();

  int dp_degree = cluster->num_total_device / model_config.ne_tp_dg;

  scheduler->initializeDummyInput(total_batch_size, 512, 512);

  scheduler->setMetadata();  // sum done
  scheduler->updateScheduler();  // after run

  scheduler->setMetadata();  // set input metadata

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
  TopModuleGraph::Ptr top1 = cluster->get_device(1)->top_module_graph;
  TopModuleGraph::Ptr top2 = cluster->get_device(2)->top_module_graph;
  TopModuleGraph::Ptr top3 = cluster->get_device(3)->top_module_graph;
  top0->print_graph();
  top1->print_graph();
  top2->print_graph();
  top3->print_graph();
  cluster->set_dependency();
  
  top0->restart_graph();
  top1->restart_graph();
  top2->restart_graph();
  top3->restart_graph();
  
  cluster->run(scheduler->getAllMetadata());
  top0->print_timeboard();
  top1->print_timeboard();
    top2->print_timeboard();
    top3->print_timeboard();

  top0->run(scheduler->running_queue[0]);
  std::cout << "-----------------------------------" << std::endl;
  top1->run(scheduler->running_queue[0]);
  std::cout << "-----------------------------------" << std::endl;
  top0->run(scheduler->running_queue[0]);
  
  return 0;
}
