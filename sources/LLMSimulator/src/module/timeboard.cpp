#include "module/timeboard.h"

#include <filesystem>
#include <fstream>

#include "module/module.h"

namespace llm_system {

TimeStamp::TimeStamp(std::string name, const StatusBoard& status,
                     TimeStamp* parent, int level)
    : name(name),
      status(status),
      start_time(start_time),
      parent(parent),
      level(level) {
  leafstamp.resize(0);
};

void TimeStamp::set_status(const StatusBoard& output_status) {
  if (output_status.parallel_execution) {
    if (output_status.processor_type == ProcessorType::GPU) {
      status.start_time = status.high_time;
      status.end_time = output_status.high_time;
    } else if (output_status.processor_type == ProcessorType::LOGIC ||
               output_status.processor_type == ProcessorType::PIM) {
      status.start_time = status.low_time;
      status.end_time = output_status.low_time;
    }
  } else {
    status.start_time = status.device_time;
    if (output_status.processor_type == ProcessorType::GPU) {
      status.end_time = output_status.high_time;
    } else if (output_status.processor_type == ProcessorType::LOGIC ||
               output_status.processor_type == ProcessorType::PIM) {
      status.end_time = output_status.low_time;
    } else {
      status.end_time = output_status.device_time;
    }
  }

  status.start_time = status.device_time;
  status.end_time = output_status.device_time;

  if (status.isTensorVec) {
    for (auto tensor : status.tensor_vec) {
      status.input_tensor_vec_shape.push_back(tensor->getShape());
    }
    for (auto tensor : output_status.tensor_vec) {
      status.output_tensor_vec_shape.push_back(tensor->getShape());
    }
  } else {
    status.input_tensor_shape = status.tensor->shape;
    status.output_tensor_shape = output_status.tensor->shape;
  }

  status.act_energy = output_status.act_energy - status.act_energy; // get layer's energy
  status.read_energy = output_status.read_energy - status.read_energy;
  status.write_energy = output_status.write_energy - status.write_energy;

  status.all_act_energy = output_status.all_act_energy - status.all_act_energy;
  status.all_read_energy = output_status.all_read_energy - status.all_read_energy;
  status.all_write_energy = output_status.all_write_energy - status.all_write_energy;
  
  status.mac_energy = output_status.mac_energy - status.mac_energy;

  status.memory_util = output_status.memory_util;
  status.compute_util = output_status.compute_util;
  status.processor_type = output_status.processor_type;
  status.flops = output_status.flops - status.flops;
  status.memory_size = output_status.memory_size - status.memory_size;

  if (status.memory_size != 0) {
    status.opb = status.flops / status.memory_size;
  }
}

void TimeStamp::find_stamp(std::string name, std::vector<TimeStamp*> &stamp_vec){
  if(this->name.find(name) != std::string::npos){
    stamp_vec.push_back(this);
  }
  else{
    for (auto& stamp : leafstamp) {
      stamp->find_stamp(name, stamp_vec);
    }
  }
}

void TimeStamp::print() {
  if (level != -1) {
    for (int i = 0; i < level; i++) {
      std::cout << "\t";
    }
    std::cout << name;
    print_time();
    print_tensor();
    print_energy();
    print_util();
    std::cout << std::endl;
  }

  for (auto& stamp : leafstamp) {
    stamp->print();
  }
}

void TimeStamp::exportGantt(std::string filepath, int device_id) {
  std::string gantt_filepath =
      filepath + "/device_" + std::to_string(device_id);

  std::ofstream ganttfile(gantt_filepath);
  std::streambuf* originalCoutBuffer = std::cout.rdbuf(ganttfile.rdbuf());

  if (level != -1) {
    for (int i = 0; i < level; i++) {
      std::cout << "\t";
    }
    std::cout << name;
    print_time();
    print_tensor();
    // print_energy();
    print_util();
    std::cout << std::endl;
  }

  for (auto& stamp : leafstamp) {
    stamp->print();
  }

  std::cout.rdbuf(originalCoutBuffer);
}

void TimeStamp::print_time() {
  time_us duration = status.end_time / 1000.0 - status.start_time / 1000.0;
  std::cout << " | " << std::setprecision(5) << duration << "us  | "
            << status.start_time / 1000.0 << " - " << status.end_time / 1000.0;
}

void TimeStamp::print_tensor() {
  if (status.isTensorVec) { // For current version, only dimneison less then 4 is implemented
    std::cout << " |";
    for (auto tensor_shape : status.input_tensor_vec_shape) {
      if(tensor_shape.size() == 2){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ")";
      }
      else if(tensor_shape.size() == 3){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);
        int i2 = tensor_shape.at(2);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ", " << std::to_string(i2) << ")";
      }
      else if(tensor_shape.size() == 4){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);
        int i2 = tensor_shape.at(2);
        int i3 = tensor_shape.at(3);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ", " << std::to_string(i2) << ", " << std::to_string(i3) << ")";
      }
    }
    std::cout << " ->";
    for (auto tensor_shape : status.output_tensor_vec_shape) {
      if(tensor_shape.size() == 2){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ")";
      }
      else if(tensor_shape.size() == 3){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);
        int i2 = tensor_shape.at(2);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ", " << std::to_string(i2) << ")";
      }
      else if(tensor_shape.size() == 4){
        int i0 = tensor_shape.at(0);
        int i1 = tensor_shape.at(1);
        int i2 = tensor_shape.at(2);
        int i3 = tensor_shape.at(3);

        std::cout << " tensor(" << std::to_string(i0) << ", "
                  << std::to_string(i1) << ", " << std::to_string(i2) << ", " << std::to_string(i3) << ")";
      }
    }
  } else {
    if(status.input_tensor_shape.size() == 2){
      int i0 = status.input_tensor_shape.at(0);
      int i1 = status.input_tensor_shape.at(1);

      int o0 = status.output_tensor_shape.at(0);
      int o1 = status.output_tensor_shape.at(1);

      std::cout << " | tensor(" << std::to_string(i0) << ", "
                << std::to_string(i1) << ") -> tensor(" << std::to_string(o0)
                << ", " << std::to_string(o1) << ")";
    }
    else if(status.input_tensor_shape.size() == 3){
      int i0 = status.input_tensor_shape.at(0);
      int i1 = status.input_tensor_shape.at(1);
      int i2 = status.input_tensor_shape.at(2);

      int o0 = status.output_tensor_shape.at(0);
      int o1 = status.output_tensor_shape.at(1);
      int o2 = status.output_tensor_shape.at(2);

      std::cout << " | tensor(" << std::to_string(i0) << ", "
                << std::to_string(i1) << ", " << std::to_string(i2) << ") -> tensor(" << std::to_string(o0)
                << ", " << std::to_string(o1) << ", " << std::to_string(o2) << ")";
    }
    else if(status.input_tensor_shape.size() == 4){
      int i0 = status.input_tensor_shape.at(0);
      int i1 = status.input_tensor_shape.at(1);
      int i2 = status.input_tensor_shape.at(2);
      int i3 = status.input_tensor_shape.at(3);

      int o0 = status.output_tensor_shape.at(0);
      int o1 = status.output_tensor_shape.at(1);
      int o2 = status.output_tensor_shape.at(2);
      int o3 = status.output_tensor_shape.at(3);

      std::cout << " | tensor(" << std::to_string(i0) << ", "
                << std::to_string(i1) << ", " << std::to_string(i2) << ", " << std::to_string(i3) << ") -> tensor(" << std::to_string(o0)
                << ", " << std::to_string(o1) << ", " << std::to_string(o2) << ", " << std::to_string(o3) << ")";
    }
  }
}

void TimeStamp::print_util() {

  std::cout << " | compute util: " << status.compute_util
            << ", memory util: " << status.memory_util << ", Op/B "
            << status.opb;
  if (status.processor_type == ProcessorType::GPU) {
    std::cout << ", GPU";
  } else if (status.processor_type == ProcessorType::LOGIC) {
    std::cout << ", Logic";
  } else if (status.processor_type == ProcessorType::PIM) {
    std::cout << ", PIM";
  }
}

void TimeStamp::print_energy() {

  std::cout << " | ACT: " << std::setprecision(5) << status.act_energy / 1000 / 1000
            << "mJ  | RD: " << std::setprecision(5) << status.read_energy / 1000 / 1000
            << "mJ  | WR: " << std::setprecision(5) << status.write_energy / 1000 / 1000
            << "mJ  | all_ACT: " << std::setprecision(5) << status.all_act_energy / 1000 / 1000 
            << "mJ  | all_RD: " << std::setprecision(5) << status.all_read_energy / 1000 / 1000
            << "mJ  | all_WR: " << std::setprecision(5) << status.all_write_energy / 1000 / 1000
            << "mJ  | MAC: " << std::setprecision(5) << status.mac_energy / 1000 / 1000
            << "mJ  | Total: " << std::setprecision(5) << (status.act_energy + status.read_energy + status.write_energy + 
            status.all_act_energy + status.all_read_energy + status.all_write_energy) / 1000 / 1000
            << "mJ";
}
}  // namespace llm_system