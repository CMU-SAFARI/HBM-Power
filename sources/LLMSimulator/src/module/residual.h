#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Residual : public Module {
    public:
        using Ptr = std::shared_ptr<Residual>;

        [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                        int hidden_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new Residual(prefix, name, hidden_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        Residual(std::string& prefix, std::string& name, int hidden_dim,
                    std::vector<int> device_list, Device::Ptr device);
        Residual() = default;

        Tensor::Ptr forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) override;
    };
}  // namespace llm_system