#include "scheduler/sequence.h"

#include "scheduler/scheduler.h"

namespace llm_system {
Sequence::Sequence(int expert_seq_id, int input_len, int output_len)
    : expert_seq_id(expert_seq_id),
      input_len(input_len),
      output_len(output_len) {
  id = count_id;
  count_id++;
  num_process_token = 0;
  current_len = 0;
  total_len = input_len + output_len - 1;
  get_expert_from_list = true;

  gen_start_time = 0;

  assertTrue(input_len > 0, "input len is not valid, input len is " +
                                std::to_string(input_len));
  assertTrue(output_len > 0, "output len is not valid, output len is " +
                                 std::to_string(output_len));

  arrival_time = 0;
  first_token_time = 0;
  end_token_time = 0;
  queueing_delay = 0;
  num_sum_iter = 0;
  sum_stage = true;
  record = false;
}

void Sequence::setGenStartTime(time_ns time) { gen_start_time = time; }

void Sequence::update(time_ns time) {
  // update end_token_time
  end_token_time += time;

  if (sum_stage) {
    first_token_time += time;
    num_sum_iter++;
  }

  // update status
  current_len += num_process_token;
  num_process_token = 0;
  if (sum_stage && current_len == input_len) {
    sum_stage = false;
  }
}

void BatchedSequence::add(Sequence::Ptr seq) {
  sequence.push_back(seq);
  return;
}

std::vector<Sequence::Ptr> BatchedSequence::get_seq() { return sequence; }

std::vector<Sequence::Ptr> BatchedSequence::get_sum() {
  std::vector<Sequence::Ptr> sum;
  sum.resize(0);
  for (auto& seq : sequence) {
    if (seq->current_len < seq->input_len) {
      sum.push_back(seq);
    }
  }
  return sum;
}

std::vector<Sequence::Ptr> BatchedSequence::get_gen() {
  std::vector<Sequence::Ptr> gen;
  for (auto& seq : sequence) {
    if (seq->current_len >= seq->input_len) {
      gen.push_back(seq);
    }
  }
  return gen;
}

void BatchedSequence::update(time_ns time) {
  for (auto& seq : sequence) {
    seq->update(time);
  }
  // cur += 32; // cur += num_layer임
}

void BatchedSequence::pop(Sequence::Ptr seq) {
  for (auto target = sequence.begin(); target != sequence.end();) {
    if ((*target)->id == seq->id) {
      target = sequence.erase(target);
    } else {
      target++;
    }
  }
}

int BatchedSequence::get_num_seq() { return sequence.size(); }

int BatchedSequence::get_process_token() {
  int token = 0;

  for (auto& seq : sequence) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_gen_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> gen_seq = get_gen();

  for (auto& seq : gen_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_sum_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> sum_seq = get_sum();

  for (auto& seq : sum_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_average_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }
  if (gen_seq.size()) {
    total_length /= gen_seq.size();
  } else {
    total_length = 0;
  }
  return total_length;
}

int BatchedSequence::get_total_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }

  return total_length;
}

void BatchedSequence::update_expert(int num_expert, int top_k,
                                    bool need_new_expert) {
  num_token_in_expert.clear();
  int size = num_token_in_expert.size();
  int num_layer =
      scheduler->model_config.num_layers / scheduler->model_config.expert_freq;

  static int expert_id = 0;

  // init
  if (size == 0) { 
    for (int i = 0; i < num_expert; i++) {
      num_token_in_expert.push_back(0);
    }
  }
  clear_expert();
  std::vector<double> skewness_weight(num_expert); // w ~ 1/k^s
  for (int i = 0 ; i < num_expert; i++){
    skewness_weight[i] = 1.0 / pow((i + 1), scheduler->model_config.skewness);
  }

  for (auto seq : sequence) {
    if (seq->num_process_token == 0) {
      continue;
    }
    int seq_id = seq->expert_seq_id;
    int current_len = seq->current_len;

    for (int token_id = current_len;
         token_id < current_len + seq->num_process_token; token_id++) {
      if (seq->get_expert_from_list) {
        for (int k = 0; k < top_k; k++) {
          int e_id = 0;
          e_id = scheduler->sequences_info[seq_id]
                     ->expert_list[token_id * num_layer + cur_layer][k];
          num_token_in_expert[e_id] += 1;
        }
      } else {
        std::set<int> expert_list;
        if(scheduler->model_config.skewness > 0){
          expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k); // for skewed expert test
        }
        else{
          expert_list = scheduler->getRandomExpert(top_k);
        }
        for (auto idx : expert_list) {
          num_token_in_expert[idx] += 1;
        }
      }
    }
  }

  cur_layer++;
  cur_layer %= num_layer;
}

void BatchedSequence::clear_expert() {
  int size = num_token_in_expert.size();
  for (int i = 0; i < size; i++) {
    num_token_in_expert.at(i) = 0;
  }
}

void BatchedSequence::add_sequence(std::vector<int> expert_seq_ids) {
  for (int seq_id : expert_seq_ids) {
    int Lin = scheduler->sequences_info.at(seq_id)->Lin;
    int Lout = scheduler->sequences_info.at(seq_id)->Lout;
    Sequence::Ptr seq = Sequence::Create(seq_id, Lin, Lout);
    add(seq);
  }
}

void BatchedSequence::add_dummy_sequence(int num_seq, int input_len,
                                         int output_len) {
  for (int i = 0; i < num_seq; i++) {
    Sequence::Ptr seq = Sequence::Create(0, input_len, output_len);
    seq->get_expert_from_list = false;
    add(seq);
  }
}

}  // namespace llm_system