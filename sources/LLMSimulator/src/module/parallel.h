#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/attention.h"
#include "module/linear.h"
#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class ColumnParallelLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<ColumnParallelLinear>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ColumnParallelLinear(prefix, name, input_size,
                                           output_size, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  ColumnParallelLinear(std::string& prefix, std::string& name, int input_size,
                       int output_size, std::vector<int> device_list,
                       Device::Ptr device);
  ColumnParallelLinear() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class RowParallelLinear : public Module {
  // Y = XA +b, X split in column, A split in row
 public:
  using Ptr = std::shared_ptr<RowParallelLinear>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new RowParallelLinear(prefix, name, input_size, output_size,
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  RowParallelLinear(std::string& prefix, std::string& name, int input_size,
                    int output_size, std::vector<int> device_list,
                    Device::Ptr device);
  RowParallelLinear() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class BatchedColumnParallelLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<BatchedColumnParallelLinear>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads, 
                                  bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new BatchedColumnParallelLinear(prefix, name, input_size, output_size,
                                  num_heads, duplicated_input, use_plain_linear, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  BatchedColumnParallelLinear(std::string& prefix, std::string& name, int input_size,
                       int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                       Device::Ptr device);
  BatchedColumnParallelLinear() = default;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
  bool duplicated_input;
  bool use_plain_linear;
};

class BatchedRowParallelLinear : public Module {
  // Y = XA +b, X split in column, A split in row
 public:
  using Ptr = std::shared_ptr<BatchedRowParallelLinear>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new BatchedRowParallelLinear(prefix, name, input_size, output_size, num_heads, 
                                        duplicated_input, use_plain_linear, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  BatchedRowParallelLinear(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                    Device::Ptr device);
  BatchedRowParallelLinear() = default;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
  bool duplicated_input;
  bool use_plain_linear;
};

class SelfAttentionParallel : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionParallel>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, bool compressed_kv,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionParallel(prefix, name, head_dim, num_heads,
                                            num_kv_heads, max_seq_len,
                                            batch_size, qk_rope_head_dim, compressed_kv, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionParallel(std::string& prefix, std::string& name, int head_dim,
                        int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                        int batch_size, bool compressed_kv, std::vector<int> device_list,
                        Device::Ptr device);
  SelfAttentionParallel() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  bool compressed_kv;
};

class RowSplit : public Module {
  // Split tensor in row
 public:
  using Ptr = std::shared_ptr<RowSplit>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new RowSplit(prefix, name, input_size, output_size, num_heads, 
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  RowSplit(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, std::vector<int> device_list,
                    Device::Ptr device);
  RowSplit() = default;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
};

class ColumnSplit : public Module {
  // Split tensor in column 
 public:
  using Ptr = std::shared_ptr<ColumnSplit>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ColumnSplit(prefix, name, input_size, output_size, num_heads, 
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  ColumnSplit(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, std::vector<int> device_list,
                    Device::Ptr device);
  ColumnSplit() = default;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
};

class MultiLatentAttentionParallel : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionParallel>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, bool compressed_kv,
                                   bool use_flash_mla, bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionParallel(prefix, name, head_dim, num_heads,
                                             num_kv_heads, max_seq_len,
                                             batch_size, qk_rope_head_dim, compressed_kv, use_flash_mla,
                                             use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    MultiLatentAttentionParallel(std::string& prefix, std::string& name, int head_dim,
                         int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                         int batch_size, bool compressed_kv, bool use_flash_mla, bool use_flash_attention,
                         std::vector<int> device_list, Device::Ptr device);
    MultiLatentAttentionParallel() = default;
 
   TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
};

class AbsorbMLAParallel : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLAParallel>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                                   bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLAParallel(prefix, name, head_dim, num_heads,
                                             num_kv_heads, max_seq_len,
                                             batch_size, qk_rope_head_dim, kv_lora_rank, compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    AbsorbMLAParallel(std::string& prefix, std::string& name, int head_dim,
                         int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                         int batch_size, int kv_lora_rank, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                         Device::Ptr device);
    AbsorbMLAParallel() = default;
 
   TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
};

}  // namespace llm_system