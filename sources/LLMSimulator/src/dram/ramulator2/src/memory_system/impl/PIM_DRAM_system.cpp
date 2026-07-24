#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"
#include "translation/translation.h"

namespace Ramulator {

class PIMDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, PIMDRAMSystem,
                                    "PIMDRAMSystem",
                                    "A PIM DRAM-based memory system.");

 protected:
  IDRAM* m_dram;
  IAddrMapper* m_addr_mapper;
  std::vector<IDRAMController*> m_controllers;

 public:
  int s_num_read_requests = 0;
  int s_num_write_requests = 0;
  int s_num_other_requests = 0;

 public:
  void init() override {
    // Create device (a top-level node wrapping all channel nodes)
    m_dram = create_child_ifce<IDRAM>();
    m_addr_mapper = create_child_ifce<IAddrMapper>();

    int num_channels = m_dram->get_level_size("channel");

    // Create memory controllers
    for (int i = 0; i < num_channels; i++) {
      IDRAMController* controller = create_child_ifce<IDRAMController>();
      controller->m_impl->set_id(fmt::format("Channel {}", i));
      controller->m_channel_id = i;
      m_controllers.push_back(controller);
    }

    m_clock_ratio = param<uint>("clock_ratio").required();

    register_stat(m_clk).name("memory_system_cycles");
    register_stat(s_num_read_requests).name("total_num_read_requests");
    register_stat(s_num_write_requests).name("total_num_write_requests");
    register_stat(s_num_other_requests).name("total_num_other_requests");
  };

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {}

  // deprecated, not work
  void setPIMmode(bool is_pim_mode) {
    for (auto controller : m_controllers) {
      controller->set_PIM_mode(is_pim_mode);
    }
  }

  bool send(Request req) override {
    m_addr_mapper->apply(req);
    int channel_id = req.addr_vec[0];
    bool is_success = m_controllers[channel_id]->send(req);
    return is_success;
  };

  void tick() override {
    m_clk++;
    m_dram->tick();
    setPIMmode(true);
    for (auto controller : m_controllers) {
      controller->tick();
    }
  };

  float get_tCK() override { return m_dram->m_timing_vals("tCK_ps") / 1000.0f; }

  bool is_finished() override {
    for (auto controller : m_controllers) {
      if (!controller->is_finished()) return false;
    }
    return true;
  }

  // const SpecDef& get_supported_requests() override {
  //   return m_dram->m_requests;
  // };
};

}  // namespace Ramulator
