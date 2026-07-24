#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_object.h"

template class std::unordered_map<llm_system::PIMOperandType,
                                  std::vector<llm_system::MemoryObject::Ptr>>;

namespace llm_system {

class DRAMRequest : public std::enable_shared_from_this<DRAMRequest> {
 public:
  friend class DRAMInterface;
  using Ptr = std::shared_ptr<DRAMRequest>;

  using PIM_Operand =
      std::unordered_map<PIMOperandType, std::vector<MemoryObject::Ptr>>;

  [[nodiscard]] static Ptr Create(DRAMRequestType type) {
    return Ptr(new DRAMRequest(type));
  }

  Ptr GetPtr() {
    return shared_from_this();
  }

  void AddOperand(const MemoryObject::Ptr arg, PIMOperandType type) {
    if (operands_.find(type) == operands_.end()) {
      operands_.emplace(type, std::vector<MemoryObject::Ptr>{});
    }
    operands_.at(type).push_back(arg);
  }

  DRAMRequestType GetType() const { return type_; }

  ~DRAMRequest() = default;

  PIM_Operand operands_;

 private:
  // Duration of DRAMRequests are not determined at the moment of construction
  DRAMRequest() = default;
  DRAMRequest(DRAMRequestType type);

  DRAMRequestType type_;
};

}  // namespace llm_system
