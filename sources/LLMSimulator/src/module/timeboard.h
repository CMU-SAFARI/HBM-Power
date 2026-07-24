#pragma once
#include <iomanip>
#include <vector>

#include "common/assert.h"
#include "common/type.h"
#include "iostream"
#include "module/status.h"

namespace llm_system {

class TimeStamp {
 public:
  // TimeStamp() = default;
  TimeStamp(std::string name = "", const StatusBoard& status = StatusBoard(),
            TimeStamp* parent = nullptr, int level = -1);

  TimeStamp* push_stamp(std::string name, const StatusBoard& status) {
    TimeStamp* stamp = new TimeStamp(name, status, this, level + 1);
    leafstamp.push_back(stamp);
    return stamp;
  }

  TimeStamp* pop_stamp(const StatusBoard& output_status) {
    set_status(output_status);
    return parent;
  }
  void find_stamp(std::string name, std::vector<TimeStamp*> &stamp_vec);

  void set_status(const StatusBoard& output_status);

  time_ns get_duration(){
    return status.end_time - status.start_time;
  }

  energy_nJ getDramEnergy(){
    return status.act_energy + status.read_energy + status.write_energy +
    status.all_act_energy + status.all_read_energy + status.all_write_energy;
  }

  energy_nJ getDramEnergyForLoad(){
    return status.act_energy_load + status.read_energy_load + status.write_energy_load +
    status.all_act_energy_load + status.all_read_energy_load + status.all_write_energy_load;
  }

  energy_nJ getCompEnergy(){
    return status.mac_energy;
  }

  double getOpb() { return status.opb; }

  void print();
  void exportGantt(std::string filepath, int device_id);

  void print_time();
  void print_tensor();
  void print_util();
  void print_energy();

  void deleteTimeStamp() {
    for (auto& stamp : leafstamp) {
      delete stamp;
    }
    leafstamp.clear();
  }

  ~TimeStamp() {
    for (auto& stamp : leafstamp) {
      delete stamp;
    }
    leafstamp.clear();
  }

 private:
  int level = -1;
  std::string name;

  time_ns start_time;
  time_ns end_time;

  StatusBoard status;

  std::vector<TimeStamp*> leafstamp;
  TimeStamp* parent;
};

class TimeBoard {
 public:
  TimeBoard() : top() { current_stamp = &top; }

  TimeBoard(const TimeBoard& copy) = delete;

  void push_timestamp(const StatusBoard& status, std::string name) {
    assertTrue(current_stamp, "No stamp in TimeBoard");
    current_stamp = current_stamp->push_stamp(name, status);
  };

  void pop_timestamp(const StatusBoard& status) {
    current_stamp = current_stamp->pop_stamp(status);
  }

  void reset_timeboard() {
    top.deleteTimeStamp();
    // print();
    //  top = TimeStamp();
    //   status = StatusBoard();
    current_stamp = &top;
  }

  // void set_time(time_ns time) { status.time = time; }
  // void add_time(time_ns time) { status.time += time; }

  void print() { top.print(); }

  void exportGantt(std::string filepath, int device_id) {
    top.exportGantt(filepath, device_id);
  };

  void find_stamp(std::string name, std::vector<TimeStamp*> &stamp_vec){
    top.find_stamp(name, stamp_vec);
  }

  // time_ns get_time() { return status.time; }

 private:
  TimeStamp top;
  // StatusBoard status;
  TimeStamp* current_stamp;
};

}  // namespace llm_system