add_library(tender polynomial.cpp index_space.cpp expr.cpp render.cpp derivation.cpp identity.cpp egraph.cpp identities.cpp)
add_library(tender::tender ALIAS tender)

target_include_directories(tender PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tender PUBLIC cxx_std_20)

set_target_properties(tender PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tender PUBLIC mpk_mix::headers)

target_compile_options(tender PRIVATE -Wall -Wextra -Werror)

if(TENDER_BUILD_COVERAGE)
    tender_enable_coverage(tender)
endif()
