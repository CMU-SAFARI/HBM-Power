#include "module/timeboard.h"

using namespace llm_system;

int main() {
  TimeBoard timeboard;
  StatusBoard status;
  timeboard.push_timestamp(status, "level1");
  timeboard.push_timestamp(status, "level2");
  timeboard.pop_timestamp(status);
  timeboard.push_timestamp(status, "level3");

  timeboard.print();
}
