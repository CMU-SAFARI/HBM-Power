#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

// A refresh manager that does nothing: no REF commands are ever issued, so the
// simulation (and the command trace) contains no refresh activity at all.
// Select with `RefreshManager: { impl: NoRefresh }` in the DRAM config.
class NoRefresh : public IRefreshManager, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, NoRefresh, "NoRefresh",
                                    "Disables DRAM refresh entirely.")
 public:
  void init() override {};
  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {};
  void tick() override {};
};

}  // namespace Ramulator
