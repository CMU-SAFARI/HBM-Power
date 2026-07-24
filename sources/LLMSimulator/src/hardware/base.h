#pragma once

#include <tuple>
#include <vector>
#include "dram/dram_type.h"

namespace llm_system {

enum class ProcessorType { NONE = 0, PIM, LOGIC, GPU, MAX };

struct LayerInfo {
  std::vector<ProcessorType> processor_type;
  int attention_group_size = 8;
  int num_heads = 64;
  int num_kv_heads = 8;
  int head_dim = 1;
  int qk_rope_head_dim = 0;
  int kv_lora_rank = 0;
  int parallel_num = 0;
  int decoder_idx = 0;
  bool parallel_execution = false;
  bool duplicated_input = false;
  bool use_flash_mla = true;
  bool use_flash_attention = true;
};

enum class LayerType {
  LINEAR = 0,
  BATCHED_LINEAR,
  ACTIVATION,
  ATTENTION_GEN,
  ATTENTION_SUM,
  ATTENTION_MIXED,
  MLA_GEN,
  MLA_SUM,
  MLA_MIXED,
  ABSORBED_MLA_GEN,
  ABSORBED_MLA_SUM,
  MAX
};

// forward declaration
class DRAMRequest;
class Tensor;
class Device;
class TopModuleGraph;
class Cluster;
class Module;
class DRAMInterface;

class MMapController;
class Device;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;
using Device_Ptr = std::shared_ptr<Device>;

using TopModuleGraph_ptr = std::shared_ptr<TopModuleGraph>;
using Module_ptr = std::shared_ptr<Module>;
using Cluster_ptr = std::shared_ptr<Cluster>;
using DRAMInterface_Ptr = std::shared_ptr<DRAMInterface>;

using MMapController_Ptr = std::shared_ptr<MMapController>;
using Device_Ptr = std::shared_ptr<Device>;

using CacheKey = std::tuple<LayerType, ProcessorType, DRAMRequestType, long>;

}  // namespace llm_system