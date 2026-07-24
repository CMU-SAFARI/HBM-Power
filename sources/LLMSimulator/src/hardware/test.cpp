#include <hardware/cluster.h>

using namespace llm_system;

int main() {
  SystemConfig config;
  Cluster::Ptr cluster = Cluster::Create(config, nullptr);

  return 0;
}