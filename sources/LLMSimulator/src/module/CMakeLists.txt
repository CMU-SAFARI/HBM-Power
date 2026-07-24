add_library(module OBJECT)

target_compile_definitions(module PRIVATE DEBUG)

target_sources(
    module PRIVATE

    module.cpp
    tensor.cpp
    module_graph.cpp
    timeboard.cpp
    activation.cpp
    layer.cpp
    layernorm.cpp
    parallel.cpp    
    linear.cpp
    attention.cpp
    compressed_kv_restore.cpp
    decoder.cpp
    communication.cpp
    embedding.cpp
    route.cpp
    rope.cpp
    residual.cpp
    expert.cpp
    lm_head.cpp
)