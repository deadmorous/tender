include(GoogleTest)

function(tender_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tender::tender GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

tender_add_test(rational_test)
tender_add_test(polynomial_test)
