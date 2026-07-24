#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "hardware/base.h"

namespace llm_system {

class Tensor;
using Tensor_Ptr = std::shared_ptr<Tensor>;

using TensorVec = std::vector<Tensor_Ptr>;

// sequence idx, kv_head idx
using kvCacheKey = std::tuple<int, int>;

}  // namespace llm_system