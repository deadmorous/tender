# Coverage helpers for GCC --coverage instrumentation.
# Included only when TENDER_BUILD_COVERAGE=ON.

# Apply --coverage flags to a target.
function(tender_enable_coverage target)
    target_compile_options(${target} PRIVATE
        --coverage
        -O0
        -fprofile-abs-path)
    target_link_options(${target} PRIVATE --coverage)
endfunction()

# Add a 'coverage' target that resets counters, runs all tests, and
# generates a gcovr HTML + text report. Fails if line coverage < 90%.
function(tender_add_coverage_target)
    find_program(GCOVR_EXECUTABLE gcovr REQUIRED
        DOC "gcovr — Python coverage report tool (pip install gcovr or apt install gcovr)")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/coverage)

    add_custom_target(coverage
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${GCOVR_EXECUTABLE}
            --root            ${CMAKE_SOURCE_DIR}
            --filter          ${CMAKE_SOURCE_DIR}/src
            --gcov-executable gcov-14
            --html-details    ${CMAKE_BINARY_DIR}/coverage/index.html
            --txt             ${CMAKE_BINARY_DIR}/coverage/summary.txt
            --print-summary
            --fail-under-line 90
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Coverage report → ${CMAKE_BINARY_DIR}/coverage/index.html"
        VERBATIM)
endfunction()
