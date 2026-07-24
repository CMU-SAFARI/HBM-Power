#include "dram/dram_request.h"

#include "common/assert.h"

namespace llm_system {

// Duration of DRAMRequests are not determined at the moment of construction
DRAMRequest::DRAMRequest(DRAMRequestType type) : type_(type) {}

}  // namespace llm_system