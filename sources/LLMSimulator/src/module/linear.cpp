
#include "linear.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// Linear //

namespace llm_system {

Linear::Linear(std::string& prefix, std::string& name, int input_size,
               int output_size, std::vector<int> device_list,
               Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {input_size, output_size};

  Tensor::Ptr A = Tensor::Create("A", shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr Y = Tensor::Create("Y", shape, "act", device, device->model_config.precision_byte);
  add_tensor(A);
  add_tensor(Y);
}

Tensor::Ptr Linear::forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr A = tensor_list.at("A");

  int m = input->shape[0];
  int k = input->shape[1];

  int n = A->shape[1];

  assertTrue(k == A->shape[0],
             module_map_name +
                 ": Dimension of input tensor for Linear is not "
                 "matched, expected " +
                 std::to_string(A->shape[0]) + " but " + std::to_string(k));

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Y", shape);
  output->perform_with_optimal = input->perform_with_optimal;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.push_back(input);
  tensor_list.push_back(A);
  tensor_list.push_back(output);

  // device->execution(total_flops);

  LayerInfo info;
  info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    info.parallel_execution = true;
    if (input->isPerformHigh()) {
      info.processor_type = {device->config.high_processor_type};
      output->setPerformHigh();
    } else {
      info.processor_type = {device->config.low_processor_type};
      output->setPerformLow();
    }
  } else if (device->config.hetero_subbatch) {
    info.processor_type = {device->config.high_processor_type};
  } else if (device->config.processor_type.size() != 1) {
    if (!input->perform_with_optimal){
      info.processor_type = {device->config.high_processor_type};
    }
  }

  device->execution(LayerType::LINEAR, tensor_list, sequences_metadata, info);

  return output;
}

BatchedLinear::BatchedLinear(std::string& prefix, std::string& name, int num_batched_gemm, int input_size,
               int output_size, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
               Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
    duplicated_input(duplicated_input),
    use_plain_linear(use_plain_linear) {
  if(!use_plain_linear){
    std::vector<int> shape = {num_batched_gemm, input_size, output_size};
    Tensor::Ptr batched_linear_input = Tensor::Create("batched_linear_input", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr batched_linear_output = Tensor::Create("batched_linear_output", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr weight = Tensor::Create("weight", shape, "weight", device, device->model_config.precision_byte);

    shape = {input_size, output_size};
    Tensor::Ptr output = Tensor::Create("output", shape, "act", device, device->model_config.precision_byte);

    add_tensor(batched_linear_input);
    add_tensor(batched_linear_output);
    add_tensor(weight);
    add_tensor(output);
  }
  else{
    std::vector<int> shape = {input_size, output_size};
    Tensor::Ptr plain_linear_input = Tensor::Create("plain_linear_input", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr plain_linear_output = Tensor::Create("plain_linear_output", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr weight = Tensor::Create("weight", shape, "weight", device, device->model_config.precision_byte);

    add_tensor(plain_linear_input);
    add_tensor(plain_linear_output);
    add_tensor(weight);
  }
}

TensorVec BatchedLinear::forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr weight = tensor_list.at("weight");
  TensorVec output_vec;

  if(!use_plain_linear){
    int num_batched_gemm = input.size();
    int m = input[0]->shape[0];
    int k = input[0]->shape[1];
    int n = weight->shape[2];

    Tensor::Ptr batched_linear_input = get_activation("batched_linear_input", {num_batched_gemm, m, k});
    Tensor::Ptr batched_linear_output = get_activation("batched_linear_output", {num_batched_gemm, m, n});

    assertTrue(k == weight->shape[1],
              module_map_name +
                  ": Dimension of input tensor for Linear is not "
                  "matched, expected " +
                  std::to_string(weight->shape[1]) + " but " + std::to_string(k));

    std::vector<int> shape = {m, n};
    Tensor::Ptr output = get_activation("output", shape);

    std::vector<Tensor::Ptr> tensor_list;
    tensor_list.push_back(batched_linear_input);
    tensor_list.push_back(weight);
    tensor_list.push_back(batched_linear_output);

    LayerInfo info;
    info.processor_type = device->config.processor_type;
    if (input[0]->parallel_execution) {
      info.parallel_execution = true;
      if (input[0]->isPerformHigh()) {
        info.processor_type = {device->config.high_processor_type};
        output->setPerformHigh();
      } else {
        info.processor_type = {device->config.low_processor_type};
        output->setPerformLow();
      }
    } else if (device->config.processor_type.size() != 1) {
      if (!input[0]->perform_with_optimal){
        info.processor_type = {device->config.high_processor_type};
      }
    }
    

    info.duplicated_input = duplicated_input;
    
    for(int bg = 0; bg < num_batched_gemm; bg++){
      output_vec.push_back(output);
    }
    
    device->execution(LayerType::BATCHED_LINEAR, tensor_list, sequences_metadata, info);
  }
  else{ // plain_linear
    int num_heads = input.size();
    int batch_size = input[0]->shape[0];
    int col = input[0]->shape[1] * num_heads;
    int weight_row = weight->shape[0];
    int weight_col = weight->shape[1];

    Tensor::Ptr plain_linear_input = get_activation("plain_linear_input", {batch_size, col});
    Tensor::Ptr plain_linear_output = get_activation("plain_linear_output", {batch_size, weight_col});

    assertTrue(col == weight_row,
              module_map_name +
                  ": Dimension of input tensor for Linear is not "
                  "matched, expected " +
                  std::to_string(weight_row) + " but " + std::to_string(col));

    std::vector<Tensor::Ptr> tensor_list;
    tensor_list.push_back(plain_linear_input);
    tensor_list.push_back(weight);
    tensor_list.push_back(plain_linear_output);

    LayerInfo info;
    info.processor_type = device->config.processor_type;
    if (input[0]->parallel_execution) {
      info.parallel_execution = true;
      if (input[0]->isPerformHigh()) {
        info.processor_type = {device->config.high_processor_type};
        plain_linear_output->setPerformHigh();
      } else {
        info.processor_type = {device->config.low_processor_type};
        plain_linear_output->setPerformLow();
      }
    } else if (device->config.processor_type.size() != 1) {
      if (!input[0]->perform_with_optimal){
        info.processor_type = {device->config.high_processor_type};
      }
    }
    

    info.duplicated_input = duplicated_input;
    
    output_vec.push_back(plain_linear_output);
    
    device->execution(LayerType::LINEAR, tensor_list, sequences_metadata, info);
  }

  return output_vec;
}

}  // namespace llm_system