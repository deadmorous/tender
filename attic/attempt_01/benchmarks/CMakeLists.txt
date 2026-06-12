function(attempt01_add_benchmark name)
    add_executable(${TENDER_A01_PREFIX}${name} ${name}.cpp)
    target_link_libraries(${TENDER_A01_PREFIX}${name} PRIVATE attempt01::tender)
endfunction()

attempt01_add_benchmark(rational_bench)
