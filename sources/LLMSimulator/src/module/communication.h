#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class AllReduce : public Module {
 public:
  using Ptr = std::shared_ptr<AllReduce>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllReduce(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AllReduce(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  AllReduce() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class AllGather : public Module {
 public:
  using Ptr = std::shared_ptr<AllGather>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllGather(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AllGather(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  AllGather() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};
 
class AllScatter : public Module { // or All-to-All
 public:
  using Ptr = std::shared_ptr<AllScatter>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllScatter(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AllScatter(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  AllScatter() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int parallel_num;
};

class MoEScatter : public Module {
 public:
  using Ptr = std::shared_ptr<MoEScatter>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEScatter(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEScatter(std::string& prefix, std::string& name,
             std::vector<int> device_list, Device::Ptr device);
  MoEScatter() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MoEGather : public Module {
 public:
  using Ptr = std::shared_ptr<MoEGather>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEGather(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEGather(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  MoEGather() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;
};

class Sync : public Module {
 public:
  using Ptr = std::shared_ptr<Sync>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
       Device::Ptr device);
  Sync() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class Sync__Set : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__Set>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__Set(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync__Set(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  Sync__Set() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class Sync__ : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync__(std::string& prefix, std::string& name, std::vector<int> device_list,
         Device::Ptr device);
  Sync__() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system