add_library(dram STATIC)

add_subdirectory(pimkernel)
add_subdirectory(ramulator2)

target_sources(
  dram PRIVATE

  dram_interface.cpp
  dram_request.cpp
  pim_request.cpp
  memory_object.cpp
  mmap_controller.cpp
)