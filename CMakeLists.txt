cmake_minimum_required(VERSION 3.18)
project(torch_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python (needed for torch/extension.h -> Python.h)
find_package(Python3 COMPONENTS Development REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

# Find Kompute
find_package(kompute QUIET)
if(NOT kompute_FOUND)
  # Try pkg-config or manual path
  find_path(KOMPUTE_INCLUDE_DIR kompute/Kompute.hpp
    HINTS /usr/local/include /usr/include
    PATH_SUFFIXES kompute)
  find_library(KOMPUTE_LIBRARY kompute
    HINTS /usr/local/lib /usr/lib)

  if(KOMPUTE_INCLUDE_DIR AND KOMPUTE_LIBRARY)
    message(STATUS "Found Kompute: ${KOMPUTE_LIBRARY}")
  else()
    message(STATUS "Kompute not found, building from source...")
    include(FetchContent)
    FetchContent_Declare(
      kompute
      GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
      GIT_TAG v0.9.0)
    set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(KOMPUTE_OPT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kompute)
    set(KOMPUTE_LIBRARY kompute)
    set(KOMPUTE_INCLUDE_DIR "")
  endif()
endif()

# The extension library
add_library(torch_vulkan_C SHARED
  csrc/torch_vulkan.cpp
  csrc/vulkan_context.cpp
  csrc/vulkan_allocator.cpp
  csrc/ops/mm.cpp
  csrc/vulkan_engine.cpp
)

target_include_directories(torch_vulkan_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Vulkan_INCLUDE_DIRS}
  ${KOMPUTE_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  csrc/
)

target_link_libraries(torch_vulkan_C PRIVATE Python3::Python)

target_link_libraries(torch_vulkan_C PRIVATE
  ${TORCH_LIBRARIES}
  ${Vulkan_LIBRARIES}
  ${KOMPUTE_LIBRARY}
)

# Output as _C.so for Python import
set_target_properties(torch_vulkan_C PROPERTIES
  OUTPUT_NAME "_C"
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/torch_vulkan"
)
