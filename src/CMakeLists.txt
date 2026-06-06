add_library(tender INTERFACE)
add_library(tender::tender ALIAS tender)

target_include_directories(tender INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tender INTERFACE cxx_std_20)

target_link_libraries(tender INTERFACE mpk_mix::headers)
