cmake_minimum_required(VERSION 3.25)
project(tender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Options ---

option(TENDER_BUILD_TESTS "Build test targets" ON)
option(TENDER_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(TENDER_BUILD_PYTHON "Build Python bindings (requires nanobind)" OFF)
option(TENDER_BUILD_COVERAGE "Build with GCC coverage instrumentation" OFF)
option(TENDER_BUILD_ATTEMPT_01 "Build attempt-01 targets (prefixed attempt01_)" OFF)

if(TENDER_BUILD_COVERAGE)
    include(cmake/coverage.cmake)
endif()

# --- Third-party dependencies ---

include(FetchContent)

# mpk_mix — unconditional utility library
FetchContent_Declare(
    mpk_mix
    GIT_REPOSITORY https://github.com/mpksoft/mpk_mix.git
    GIT_TAG        ad48756c35d01ccab96f1ffa4858a358e9357fff
)
set(MPKMIX_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mpk_mix)

# GoogleTest — for unit and integration tests
if(TENDER_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# nanobind — Python bindings (activated in Phase 12)
if(TENDER_BUILD_PYTHON)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.4.0
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# --- Attempt 01 (archived) ---

if(TENDER_BUILD_ATTEMPT_01)
    add_subdirectory(attic/attempt_01)
endif()

# --- tender library (attempt 02) ---

add_subdirectory(src)

# --- Tests ---

if(TENDER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(TENDER_BUILD_COVERAGE)
    tender_add_coverage_target()
endif()
