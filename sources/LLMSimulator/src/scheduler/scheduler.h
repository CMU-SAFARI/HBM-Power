#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>


#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Scheduler : public std::enable_shared_from_this<Scheduler> {
 public:
  using Ptr = std::shared_ptr<Scheduler>;

  [[nodiscard]] static Ptr Create(SystemConfig system_config,
                                  ModelConfig& model_config,
                                  std::string expert_file_path = "",
                                  int batch_size = 4096,
                                  int num_max_batched_token = 4096 * 64,
                                  int max_process_token = 8192 * 4) {
    Ptr ptr = Ptr(new Scheduler(system_config, model_config, expert_file_path,
                                batch_size, num_max_batched_token,
                                max_process_token));
    ptr->initRunningQueue();
    return ptr;
  };

  Ptr getPtr() { return shared_from_this(); }

  void clear();

  void pushSeq(int num_seq);
  void pushDummySeq(int input_len = 256, int max_len = 1024);
  void pushRealSeq(int num_seq);

  void initializeDummyInput(int num_seq, int input_len, int output_len);

  void initializeRealInput(int num_seq);

  void hittingQueue(int iter);

  bool hasSumSeq();

  std::vector<BatchedSequence::Ptr> getAllMetadata();
  BatchedSequence::Ptr getMetadata(int dp_rank);
  BatchedSequence::Ptr getMaxMetadata(int num_expert, int top_k, Ptr scheduler = nullptr);

  std::vector<BatchedSequence::Ptr> setMetadata();
  std::vector<Sequence::Ptr> updateScheduler(time_ns time = 0);
  std::vector<Sequence::Ptr> updateSchedulerSumGenSplit(time_ns time = 0);

  void printStatus();

  // random
  std::set<int> getRandomExpert(int top_k);
  std::set<int> getZipfianRandomExpert(std::vector<double> skewness_weight, int top_k);
  std::set<int> getEquallyDistributedExpert(int token_id, int top_k);
  int getRandomExpertSeqId();
  static double getNormaldistribution();
  static int getPoissondistribution(int request_per_second);
  static int getNumInjection();

  void getActualArrivalTime(int num_iter);

  void fillSequenceQueue(time_ns iter_time = 0, time_ns total_time = 0);

  // time to fill running queue
  void fillRunningQueue(time_ns time = 0);

  int getBatchSize();
  int getSumSize();
  int getGenSize();

  int getAverageSeqlen();
  int getNumProcessToken();

  int total_seq_num;

  time_ns total_time;

  int total_batch_size;
  int batch_size_per_dp;  // per dp
  int num_max_batched_token;
  int max_process_token;
  int dp_degree;

  bool real_data;
  bool real_expert_data;

  std::vector<Sequence::Ptr> sequence_queue;
  std::vector<BatchedSequence::Ptr> running_queue;
  
  std::vector<time_ns> actual_arrival_time; 
  int cur_arrival_time_idx;
  std::vector<int> total_token_in_expert;

  SystemConfig system_config;
  ModelConfig& model_config;
  std::vector<SequenceInfo::Ptr> sequences_info;
  std::string expert_file_path;
  
  void initRunningQueue(); // juhwan
  
 private:
  Scheduler(SystemConfig system_config, ModelConfig& model_config,
            std::string expert_file, int batch_size, int num_max_batched_token,
            int max_process_token);

  void initExpertList(std::string expert_file_path);
  // void initRunningQueue();

  bool disagg_system = false;
};

}  // namespace llm_system