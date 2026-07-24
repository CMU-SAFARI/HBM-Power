#pragma once
#include <map>
#include <string>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "model/llm.h"
#include "model/model_config.h"
#include "module/module.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class Model : public Module {
 public:
  Model(ModelConfig& model_config, Cluster::Ptr cluster,
        Scheduler::Ptr scheduler)
      : Module(),
        model_config(model_config),
        cluster(cluster),
        scheduler(scheduler) {
    for (int i = 0; i < cluster->num_total_device; i++) {
      device = cluster->get_device(i);
      device->setModelConfig(model_config);
      Module::Ptr top_module = std::static_pointer_cast<Module>(
          LLM::Create(model_config, cluster, scheduler, device));

      add_module(top_module);

      model_distribute(top_module, cluster, scheduler, i);
      device->reset_timeboard();
      device->reset_status();
    }
  };

 private:
  void model_distribute(Module::Ptr top_module, Cluster::Ptr cluster,
                        Scheduler::Ptr scheduler, int device_rank) {
    BatchedSequence::Ptr max_metadata = scheduler->getMaxMetadata(model_config.num_routed_expert, model_config.top_k);
    // Tensor::Ptr input_tensor = Tensor::Create(
    //     "EmbeddingVector", {1, max_metadata->get_process_token()}, "act",
    //     cluster->get_device(device_rank));


    Tensor::Ptr input_tensor = Tensor::Create(
          "EmbeddingVector", {1, scheduler->model_config.n_vocab}, "act",
          cluster->get_device(device_rank), scheduler->model_config.precision_byte);
    // Tensor::Ptr input_tensor =
    //     Tensor::Create("EmbeddingVector", {64, model_config.hidden_dim},
    //     "act",
    //                    cluster->get_device(device_rank));
    (*top_module)(input_tensor, max_metadata);
  };

  Cluster::Ptr cluster;
  Scheduler::Ptr scheduler;
  ModelConfig model_config;
};

}  // namespace llm_system