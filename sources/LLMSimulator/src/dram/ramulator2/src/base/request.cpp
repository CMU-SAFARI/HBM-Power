#include "base/request.h"

namespace Ramulator {

Request::Request(Addr_t addr, int type) : addr(addr), type_id(type) { init(); }

Request::Request(AddrVec_t addr_vec, int type)
    : addr_vec(addr_vec), type_id(type) {
  init();
}

Request::Request(AddrVec_t addr_vec, int type, llm_system::PIMCommand& pimcmd)
    : addr_vec(addr_vec), type_id(type), pimcmd(pimcmd) {
  init();
}

Request::Request(AddrVec_t addr_vec, int type, llm_system::PIMCommand& pimcmd,
                 std::function<void(Request&)> callback)
    : addr_vec(addr_vec), type_id(type), pimcmd(pimcmd), callback(callback) {
  init();
}

Request::Request(AddrVec_t addr_vec, int type, llm_system::PIMCommand& pimcmd,
  std::function<void(Request&)> callback, bool from_mem_pool)
: addr_vec(addr_vec), type_id(type), pimcmd(pimcmd), callback(callback), from_mem_pool(from_mem_pool){
init();
}

Request::Request(Addr_t addr, int type, int source_id,
                 std::function<void(Request&)> callback)
    : addr(addr), type_id(type), source_id(source_id), callback(callback) {
  init();
}

void Request::init() {
  issued_dram_cmd.resize((int)llm_system::DRAMCommandType::kMAX, 0);
}

void Request::update_state() {
  switch ((int)command) {
    case 0:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kACT]++;
      break;
    case 1:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kALL_ACT]++;
      break;
    case 4:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kALL_PRE]++;
      break;
    case 5:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kREAD]++;
      break;
    case 6:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kWRITE]++;
      break;
    case 9:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kALL_READ]++;
      break;
    case 10:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kALL_WRITE]++;
      break;
    case 11:
      issued_dram_cmd[(int)llm_system::DRAMCommandType::kREF]++;
      break;
    default:
      break;
  }
}

void Request::update_mem_pool_state() {
  switch ((int)command) {
    case 0:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kACT_1]++;
      break;
    case 1:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kACT_2]++;
      break;
    case 2:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kALL_ACT_1]++;
      break;
    case 3:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kALL_ACT_2]++;
      break;
    case 4:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kPRE]++;
      break;
    case 5:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kPREA]++;
      break;
    case 6:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kALL_PRE]++;
      break;
    case 7:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kCASRD]++;
      break;
    case 8:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kCASWR]++;
      break;
    case 9:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kRD16]++;
      break;
    case 10:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kWR16]++;
      break;
    case 11:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kRD16A]++;
      break;
    case 12:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kWR16A]++;
      break;
    case 13:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kALL_RD]++;
      break;
    case 14:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kALL_WR]++;
      break;
    case 15:
      issued_dram_cmd[(int)llm_system::LPDDR5CommandType::kREF]++;
      break;
    default:
      break;
  }
}

}  // namespace Ramulator
