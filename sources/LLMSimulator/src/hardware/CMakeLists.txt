add_library(hardware OBJECT)

target_sources(
    hardware PRIVATE

    cluster.cpp
    node.cpp
    device.cpp
    executor.cpp
    
    layer_impl.cpp
    activation_impl.cpp
    linear_impl.cpp
    attention_sum_impl.cpp
    attention_gen_impl.cpp
    attention_mixed_impl.cpp
)