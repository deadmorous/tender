# Benchmarks — standalone executables, not registered with CTest.
# Built with optimization regardless of the surrounding build type so timing is
# meaningful; the reported pass/node/class counts are deterministic either way.

function(tender_add_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tender::tender)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Werror -O2)
endfunction()

tender_add_benchmark(egraph_saturate_bench)
