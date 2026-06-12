# Attempt 01 wrapper — included by the root CMakeLists.txt when
# TENDER_BUILD_ATTEMPT_01 is ON.  All targets and output binaries produced
# here carry the "attempt01_" prefix so they are unambiguous when both
# attempts are built in the same tree.

set(TENDER_A01_PREFIX "attempt01_")

add_subdirectory(src)

if(TENDER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(TENDER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TENDER_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(TENDER_BUILD_COVERAGE)
    tender_add_coverage_target()
endif()
