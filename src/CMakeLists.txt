add_library(tender polynomial.cpp)
add_library(tender::tender ALIAS tender)

target_include_directories(tender PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tender PUBLIC cxx_std_20)

set_target_properties(tender PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tender PUBLIC mpk_mix::headers)
