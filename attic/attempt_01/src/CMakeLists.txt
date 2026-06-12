add_library(${TENDER_A01_PREFIX}tender
    basis.cpp
    coord_system.cpp
    derivation.cpp
    integral.cpp
    expr.cpp
    identity.cpp
    index.cpp
    match.cpp
    polynomial.cpp
    polynomial_expr.cpp
    simplify.cpp)
add_library(attempt01::tender ALIAS ${TENDER_A01_PREFIX}tender)

target_include_directories(${TENDER_A01_PREFIX}tender PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(${TENDER_A01_PREFIX}tender PUBLIC cxx_std_20)

set_target_properties(${TENDER_A01_PREFIX}tender PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${TENDER_A01_PREFIX}tender PUBLIC mpk_mix::headers)

if(TENDER_BUILD_COVERAGE)
    tender_enable_coverage(${TENDER_A01_PREFIX}tender)
endif()
