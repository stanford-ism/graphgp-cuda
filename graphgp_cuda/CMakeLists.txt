cmake_minimum_required(VERSION 3.18...3.27)
project(graphgp_cuda LANGUAGES CXX CUDA)
message(STATUS "Project name: graphgp_cuda")

# Find Python
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Find FFI include directory
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES native) #"89;90")

# Add the shared library target
add_library(graphgp_cuda SHARED xla_bindings.cu)
target_include_directories(graphgp_cuda PRIVATE ${XLA_DIR})
set_target_properties(graphgp_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)