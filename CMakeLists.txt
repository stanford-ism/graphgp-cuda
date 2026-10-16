cmake_minimum_required(VERSION 3.24)

# Find Python first: it locates the JAX FFI headers and, if installed, the pip CUDA toolkit
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Prefer the CUDA toolkit installed by pip (nvidia-cuda-nvcc and nvidia-cuda-runtime, also pulled in by jax[cuda13]),
# so no system toolkit is needed and the compiler matches JAX's CUDA. A compiler given explicitly with
# -DCMAKE_CUDA_COMPILER or the CUDACXX environment variable takes precedence, otherwise CMake searches the system.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from nvidia import cu13; print(next(iter(cu13.__path__)))"
  OUTPUT_VARIABLE PIP_CUDA_ROOT RESULT_VARIABLE PIP_CUDA_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(PIP_CUDA_RESULT EQUAL 0 AND NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  message(STATUS "Using pip CUDA toolkit: ${PIP_CUDA_ROOT}")
  set(CMAKE_CUDA_COMPILER "${PIP_CUDA_ROOT}/bin/nvcc")
  set(ENV{LIBRARY_PATH} "${PIP_CUDA_ROOT}/lib:$ENV{LIBRARY_PATH}") # for the compiler check during project()
  link_directories("${PIP_CUDA_ROOT}/lib")
endif()
# Link the CUDA runtime statically, so the library does not depend on finding libcudart at load time
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# CUDA architectures: override with the CUDAARCHS environment variable or -DCMAKE_CUDA_ARCHITECTURES (e.g. "native" or
# "120-real" for a quick local build). The default embeds SASS and PTX for A100=80, RTX 40xx=89, H100=90, RTX 50xx=120.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 80;89;90;120)
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)
message(STATUS "Project name: ${SKBUILD_PROJECT_NAME}, CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")

# Find FFI include directory
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR RESULT_VARIABLE XLA_RESULT)
if(NOT XLA_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not find the XLA FFI include directory via jax")
endif()
message(STATUS "XLA include directory: ${XLA_DIR}")

# Development builds compile only a few (k, n_dim) template instantiations, for much faster compilation
option(GRAPHGP_CUDA_DEV_BUILD "Compile only k <= 16 and n_dim in {2, 3}" OFF)

# Add the shared library target
add_library(${SKBUILD_PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME}/xla_bindings.cu)
target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME} ${XLA_DIR})
set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(GRAPHGP_CUDA_DEV_BUILD)
  target_compile_definitions(${SKBUILD_PROJECT_NAME} PRIVATE GRAPHGP_CUDA_DEV_BUILD)
endif()
install(TARGETS ${SKBUILD_PROJECT_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
