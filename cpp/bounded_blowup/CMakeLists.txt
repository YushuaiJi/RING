# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(rfbb_bounded_blowup_cpp LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RFBB_BUILD_TESTS "Build the standalone native correctness tests" ON)
option(RFBB_BUILD_PYTHON "Build the pybind11 extension" ON)
option(RFBB_ENABLE_NATIVE_ARCH
       "Enable -march=native -mtune=native for the frozen host build" OFF)
set(RFBB_CPP_SOURCE_TREE_SHA256 "UNSET" CACHE STRING
    "SHA-256 of the frozen cpp/bounded_blowup source tree")
set(RFBB_NUMERIC_FLAGS -ffp-contract=off -fno-fast-math)

add_library(rfbb_core STATIC
  src/digest.cpp
  src/verified_corpus.cpp
  src/construction.cpp
  src/leaf_build.cpp
  src/leaf_repair.cpp
  src/logical_graph.cpp
  src/logical_artifact.cpp
  src/routing_layout.cpp
  src/logical_search.cpp
  src/native_bulk_build.cpp
  src/router.cpp
  src/search.cpp
  src/validation.cpp
  src/hnsw_baseline.cpp
)
target_include_directories(rfbb_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rfbb_core PUBLIC Threads::Threads)
target_compile_options(rfbb_core PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  ${RFBB_NUMERIC_FLAGS}
)
target_compile_definitions(rfbb_core PRIVATE RFBB_NUMERIC_INTERNAL=1)
# GCC 8 keeps std::filesystem in a separate compatibility library.  Without
# this propagated link dependency the extension links with unresolved symbols
# and fails only when Python imports the resulting shared object.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(rfbb_core PUBLIC stdc++fs)
endif()
if(RFBB_ENABLE_NATIVE_ARCH)
  target_compile_options(rfbb_core PRIVATE -march=native -mtune=native)
endif()

if(RFBB_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_bounded_blowup_cpp MODULE src/python_module.cpp)
  target_link_libraries(_bounded_blowup_cpp PRIVATE rfbb_core)
  target_compile_options(_bounded_blowup_cpp PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    ${RFBB_NUMERIC_FLAGS}
  )
  target_compile_definitions(_bounded_blowup_cpp PRIVATE
    RFBB_CPP_SOURCE_TREE_SHA256="${RFBB_CPP_SOURCE_TREE_SHA256}"
  )
  if(RFBB_ENABLE_NATIVE_ARCH)
    target_compile_options(_bounded_blowup_cpp PRIVATE
      -march=native -mtune=native)
  endif()

  install(TARGETS _bounded_blowup_cpp
    LIBRARY DESTINATION mr_ann/bounded_block_graph
  )
endif()

if(RFBB_BUILD_TESTS)
  enable_testing()
  foreach(test_name IN ITEMS test_router test_template_router test_search
                               test_malformed_index test_construction
                               test_leaf_build test_leaf_repair
                               test_logical_graph test_digest
                               test_verified_corpus)
    if("${test_name}" STREQUAL "test_verified_corpus")
      # Keep the fixture-only seam out of the production core (and any
      # pybind-linked binary); this target is a standalone test variant.
      add_executable(${test_name}
        tests/${test_name}.cpp
        src/digest.cpp
        src/verified_corpus.cpp)
      target_link_libraries(${test_name} PRIVATE Threads::Threads)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
         CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(${test_name} PRIVATE stdc++fs)
      endif()
    elseif("${test_name}" STREQUAL "test_leaf_build")
      # Keep the worker-launch counter test-only: compiling leaf_build.cpp
      # directly here prevents RFBB_LEAF_BUILD_TESTING from entering
      # rfbb_core or any production-linked target.
      add_executable(${test_name}
        tests/${test_name}.cpp
        src/leaf_build.cpp)
      target_link_libraries(${test_name} PRIVATE Threads::Threads)
    else()
      add_executable(${test_name} tests/${test_name}.cpp)
      target_link_libraries(${test_name} PRIVATE rfbb_core)
    endif()
    target_include_directories(${test_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${test_name} PRIVATE RFBB_NUMERIC_INTERNAL=1)
    if("${test_name}" STREQUAL "test_leaf_build")
      target_compile_definitions(${test_name} PRIVATE
        RFBB_LEAF_BUILD_TESTING=1)
    endif()
    if("${test_name}" STREQUAL "test_verified_corpus")
      target_compile_definitions(${test_name} PRIVATE
        RFBB_VERIFIED_CORPUS_TESTING=1)
    endif()
    target_compile_options(${test_name} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow
      ${RFBB_NUMERIC_FLAGS}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  # Artifact V3 tests need the fixture-only verified-corpus opener.  Compile a
  # standalone test variant so the production core and Python module never
  # contain that seam.
  add_executable(test_logical_artifact
    tests/test_logical_artifact.cpp
    src/digest.cpp
    src/verified_corpus.cpp
    src/construction.cpp
    src/leaf_build.cpp
    src/leaf_repair.cpp
    src/logical_graph.cpp
    src/logical_artifact.cpp)
    # routing_layout.cpp is part of the V3 artifact codec used by this target.
  target_sources(test_logical_artifact PRIVATE src/routing_layout.cpp)
  target_include_directories(test_logical_artifact PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_logical_artifact PRIVATE Threads::Threads)
  target_compile_definitions(test_logical_artifact PRIVATE
    RFBB_NUMERIC_INTERNAL=1
    RFBB_VERIFIED_CORPUS_TESTING=1)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(test_logical_artifact PRIVATE stdc++fs)
  endif()
  target_compile_options(test_logical_artifact PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    ${RFBB_NUMERIC_FLAGS}
  )
  add_test(NAME test_logical_artifact COMMAND test_logical_artifact)

  # Logical query tests use the fixture-only verified-corpus opener just like
  # Artifact V3 tests.  Keep this target standalone so the testing seam (and
  # its macro) cannot leak into rfbb_core or the Python extension.
  add_executable(test_logical_search
    tests/test_logical_search.cpp
    src/digest.cpp
    src/verified_corpus.cpp
    src/construction.cpp
    src/leaf_build.cpp
    src/leaf_repair.cpp
    src/logical_graph.cpp
    src/logical_artifact.cpp
    src/routing_layout.cpp
    src/logical_search.cpp
    src/router.cpp
    src/search.cpp
    src/validation.cpp)
  target_include_directories(test_logical_search PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_logical_search PRIVATE Threads::Threads)
  target_compile_definitions(test_logical_search PRIVATE
    RFBB_NUMERIC_INTERNAL=1
    RFBB_VERIFIED_CORPUS_TESTING=1)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(test_logical_search PRIVATE stdc++fs)
  endif()
  target_compile_options(test_logical_search PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    ${RFBB_NUMERIC_FLAGS}
  )
  add_test(NAME test_logical_search COMMAND test_logical_search)

  # Stage-A native bulk tests use the fixture-only corpus opener.  Keep the
  # testing macro confined to this standalone target; rfbb_core and the
  # Python extension remain production-clean.
  add_executable(test_native_bulk
    tests/test_native_bulk_build.cpp
    src/digest.cpp
    src/verified_corpus.cpp
    src/construction.cpp
    src/leaf_build.cpp
    src/leaf_repair.cpp
    src/logical_graph.cpp
    src/logical_artifact.cpp
    src/routing_layout.cpp
    src/logical_search.cpp
    src/native_bulk_build.cpp
    src/router.cpp
    src/search.cpp
    src/validation.cpp)
  target_include_directories(test_native_bulk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_native_bulk PRIVATE Threads::Threads)
  target_compile_definitions(test_native_bulk PRIVATE
    RFBB_NUMERIC_INTERNAL=1
    RFBB_VERIFIED_CORPUS_TESTING=1)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(test_native_bulk PRIVATE stdc++fs)
  endif()
  target_compile_options(test_native_bulk PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    ${RFBB_NUMERIC_FLAGS}
  )
  add_test(NAME test_native_bulk COMMAND test_native_bulk)
endif()
