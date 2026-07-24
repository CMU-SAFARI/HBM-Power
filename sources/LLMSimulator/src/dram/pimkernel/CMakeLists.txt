#add_library(pimkernel OBJECT)

target_sources(
  dram PRIVATE

  pim_kernel.cpp
  Read.cpp
  Write.cpp
  GEMV.cpp
)



