#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class LayerNorm : public Module {
    public:
        using Ptr = std::shared_ptr<LayerNorm>;

        [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                        int hidden_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new LayerNorm(prefix, name, hidden_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        LayerNorm(std::string& prefix, std::string& name, int hidden_dim,
                    std::vector<int> device_list, Device::Ptr device);
        LayerNorm() = default;

        Tensor::Ptr forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) override;
    };
}  // namespace llm_system