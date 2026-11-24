# FILENAME: CMakeLists.txt

cmake_minimum_required(VERSION 3.25)
project(
  theta-hyper-shared-ptr
  VERSION 0.2.0
  LANGUAGES CXX)

find_package(theta-cmake CONFIG QUIET)
if(NOT theta-cmake_FOUND
   AND EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/cmake/ThetaBuild.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/cmake/ThetaBuild.cmake")
endif()

theta_enable_strict_warnings()
theta_add_header_check()

find_package(Threads REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_compile_options(-O3 -march=armv8.2-a+lse+rcpc)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_compile_options(-fomit-frame-pointer -fno-stack-protector)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-O3 -mbranches-within-32B-boundaries)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_compile_options(-fomit-frame-pointer -fno-stack-protector)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND THETA_ENABLE_LTO)
    add_link_options(-Wl,-mllvm,--x86-branches-within-32B-boundaries)
  endif()
else()
  add_compile_options(-O3)
endif()

add_library(theta-hyper-shared-ptr STATIC src/hyper_shared_ptr.cpp)
add_library(theta::theta-hyper-shared-ptr ALIAS theta-hyper-shared-ptr)
add_library(theta::hyper-shared-ptr ALIAS theta-hyper-shared-ptr)
set_target_properties(theta-hyper-shared-ptr
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  theta-hyper-shared-ptr
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(theta-hyper-shared-ptr PUBLIC Threads::Threads)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  # Separate object library compiled WITHOUT Theta headers in scope to prove
  # binary ABI compatibility across unmodified std::shared_ptr / std::weak_ptr
  # TUs.
  add_library(opaque_std_interop OBJECT tests/opaque_std_interop.cpp)

  add_executable(hyper_shared_ptr_tests tests/test_hyper_shared_ptr.cpp
                                        $<TARGET_OBJECTS:opaque_std_interop>)
  target_link_libraries(hyper_shared_ptr_tests
                        PRIVATE theta::theta-hyper-shared-ptr GTest::gtest_main)
  gtest_discover_tests(
    hyper_shared_ptr_tests DISCOVERY_TIMEOUT 10
    PROPERTIES
    TIMEOUT 30)

  add_executable(hyper_shared_ptr_hardened_tests tests/test_hardened.cpp)
  target_link_libraries(hyper_shared_ptr_hardened_tests
                        PRIVATE theta::theta-hyper-shared-ptr GTest::gtest_main)
  gtest_discover_tests(
    hyper_shared_ptr_hardened_tests DISCOVERY_TIMEOUT 10
    PROPERTIES
    TIMEOUT 30)
endif()

if(BUILD_BENCHMARK
   OR "$ENV{BUILD_BENCHMARK}" STREQUAL "ON"
   OR "$ENV{BUILD_BENCHMARK}" STREQUAL "true")
  find_package(benchmark REQUIRED)

  add_executable(hyper_shared_ptr_bench benchmarks/bench_hyper_shared_ptr.cpp)
  target_link_libraries(
    hyper_shared_ptr_bench PRIVATE theta::theta-hyper-shared-ptr
                                   benchmark::benchmark Threads::Threads)
endif()

theta_install_package(
  TARGETS
  theta-hyper-shared-ptr
  FIND_DEPENDENCIES
  Threads
  DESCRIPTION
  "8192-CPU rseq std::shared_ptr ABI-compatible and handle-tagged concurrent smart pointers"
)
