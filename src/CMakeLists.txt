add_library(tender
    expr.cpp
    index.cpp
    polynomial.cpp
    polynomial_expr.cpp)
add_library(tender::tender ALIAS tender)

target_include_directories(tender PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tender PUBLIC cxx_std_20)

target_link_libraries(tender PUBLIC mpk_mix::headers)

if(TENDER_BUILD_COVERAGE)
    tender_enable_coverage(tender)
endif()
