cmake_minimum_required(VERSION 3.18...3.27)
project(graphgp_cuda LANGUAGES CXX CUDA)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE XLA_DIR
)

message(STATUS "XLA include directory: ${XLA_DIR}")

# NVIDIA T550 needs sm_75. Keep the others for portability.
set(CMAKE_CUDA_ARCHITECTURES 75;80;89;90)

add_library(graphgp_cuda SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/graphgp_cuda/xla_bindings.cu
)

target_include_directories(graphgp_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/graphgp_cuda
  ${XLA_DIR}
)

set_target_properties(graphgp_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS graphgp_cuda LIBRARY DESTINATION graphgp_cuda)
