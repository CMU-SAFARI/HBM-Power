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

class CompressedKVRestore : public Module {
    public:
     using Ptr = std::shared_ptr<CompressedKVRestore>;
     [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                     int head_dim, int num_heads, int max_seq_len, int batch_size, int kv_lora_rank, int qk_rope_head_dim,
                                     bool compressed_kv, std::vector<int> device_list,
                                     Device::Ptr device) {
       Ptr ptr =
           Ptr(new CompressedKVRestore(prefix, name, head_dim, num_heads, 
                                  max_seq_len, batch_size, kv_lora_rank, qk_rope_head_dim, compressed_kv, device_list, device));
       ptr->set_tensor_module();
       return ptr;
     };
   
    private:
     CompressedKVRestore(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int max_seq_len,
                    int batch_size, int kv_lora_rank, int qk_rope_head_dim, bool compressed_kv, std::vector<int> device_list,
                    Device::Ptr device);
     CompressedKVRestore() = default;
   
     TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
   
    private:
     int rank;
     int head_dim;
     int num_heads;
     int kv_lora_rank;
     int qk_rope_head_dim;
     int max_seq_len;
     bool compressed_kv;
   };

}  // namespace llm_system