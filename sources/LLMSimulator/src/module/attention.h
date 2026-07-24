#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class SelfAttentionGen : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionGen>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionGen(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionGen(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                   Device::Ptr device);
  SelfAttentionGen() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

class SelfAttentionSum : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionSum>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionSum(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionSum(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                   Device::Ptr device);
  SelfAttentionSum() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

class SelfAttentionMixed : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionMixed>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionMixed(prefix, name, head_dim, num_heads,
                                         num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                         device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionMixed(std::string& prefix, std::string& name, int head_dim,
                     int num_heads, int num_kv_heads, int max_seq_len,
                     int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                     Device::Ptr device);
  SelfAttentionMixed() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

class AttentionSplit : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionSplit>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionSplit(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AttentionSplit(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                 bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  AttentionSplit() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

class AttentionMerge : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionMerge>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionMerge(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AttentionMerge(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  AttentionMerge() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

class MultiLatentAttentionGen : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionGen>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    MultiLatentAttentionGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                    Device::Ptr device);
    MultiLatentAttentionGen() = default;
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
   bool use_flash_mla;
   // std::vector<int> device_list;
 };
 
 class MultiLatentAttentionSum : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionSum>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                   bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionSum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    MultiLatentAttentionSum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool use_flash_attention, 
                    std::vector<int> device_list, Device::Ptr device);
    MultiLatentAttentionSum() = default;
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
 
 class AbsorbMLAGen : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLAGen>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   int kv_lora_rank, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLAGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    AbsorbMLAGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool compressed_kv,
                    bool use_flash_mla, std::vector<int> device_list, Device::Ptr device);
    AbsorbMLAGen() = default;
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   bool compressed_kv;
   bool use_flash_mla;
   // std::vector<int> device_list;
 };
 
 class AbsorbMLASum : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLASum>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                   int kv_lora_rank, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLASum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    AbsorbMLASum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, std::vector<int> device_list,
                    Device::Ptr device);
    AbsorbMLASum() = default;
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   // std::vector<int> device_list;
 };
}  // namespace llm_system