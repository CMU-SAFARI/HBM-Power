#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class RoPE : public Module {
    public:
        using Ptr = std::shared_ptr<RoPE>;

        [[nodiscard]] static Ptr Create(std::string prefix, std::string name, int num_heads,
                                        int qk_rope_head_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new RoPE(prefix, name, num_heads, qk_rope_head_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        RoPE(std::string& prefix, std::string& name, int num_heads, int qk_rope_head_dim,
                    std::vector<int> device_list, Device::Ptr device);
        RoPE() = default;

        Tensor::Ptr forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) override;
        int qk_rope_head_dim;
};
class BatchedRoPE : public Module {
    public:
        using Ptr = std::shared_ptr<BatchedRoPE>;

        [[nodiscard]] static Ptr Create(std::string prefix, std::string name, int num_heads,
                                        int qk_rope_head_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new BatchedRoPE(prefix, name, num_heads, qk_rope_head_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        BatchedRoPE(std::string& prefix, std::string& name, int num_heads, int qk_rope_head_dim,
                    std::vector<int> device_list, Device::Ptr device);
        BatchedRoPE() = default;

        TensorVec forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) override;
        int qk_rope_head_dim;
    };
}  // namespace llm_system