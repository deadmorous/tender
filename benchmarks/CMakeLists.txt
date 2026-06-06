function(tender_add_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tender::tender)
endfunction()

tender_add_benchmark(rational_bench)
