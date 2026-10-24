cmake_minimum_required(VERSION 3.18)
project(ring LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(RING_NATIVE_ARCH "Compile with -march=native (recommended; enables AVX kernels)" ON)

# Core graph library (construction, leaf repair, graph search).
set(RFBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RFBB_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(RFBB_ENABLE_NATIVE_ARCH ${RING_NATIVE_ARCH} CACHE BOOL "" FORCE)
add_subdirectory(cpp/bounded_blowup)

# Python extension used by the ring package (module name _frans_v11).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)
set(PYBIND11_INCLUDE_DIR "" CACHE PATH "Directory containing pybind11/pybind11.h (optional)")
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_frans_v11 src/ring_native.cpp)
else()
  if(NOT PYBIND11_INCLUDE_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
                    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT EXISTS "${PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR "pybind11 not found: pip install pybind11")
  endif()
  Python_add_library(_frans_v11 MODULE WITH_SOABI src/ring_native.cpp)
  target_include_directories(_frans_v11 PRIVATE "${PYBIND11_INCLUDE_DIR}")
endif()
target_link_libraries(_frans_v11 PRIVATE rfbb_core Threads::Threads)
target_compile_options(_frans_v11 PRIVATE -O3 -ffp-contract=off -fno-fast-math)
if(RING_NATIVE_ARCH)
  target_compile_options(_frans_v11 PRIVATE -march=native -mtune=native)
endif()
# Place the extension inside the Python package so `import ring` finds it.
set_target_properties(_frans_v11 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ring")
