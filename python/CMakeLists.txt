find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
nanobind_add_module(_tender NB_STATIC _tender.cpp)

target_link_libraries(_tender PRIVATE tender::tender)

# Put the compiled extension inside the tender/ package directory so that
# PYTHONPATH=<build>/python makes "import tender" work.
set_target_properties(_tender PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tender)

# Keep Python source files in sync with the build tree at build time
# (not just at configure time like configure_file would do).
# Ensure subdirectories exist in the build tree before copying.
file(MAKE_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/tender/lib
    ${CMAKE_CURRENT_BINARY_DIR}/tender/lib/identities)

foreach(_py
    tender/__init__.py
    tender/lib/__init__.py
    tender/lib/identities/__init__.py
    tender/lib/identities/epsilon.py
    tender/lib/identities/identity_tensor.py
    tender/lib/identities/functions.py
    test_tender.py)
    add_custom_command(
        OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${_py}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/${_py}
                ${CMAKE_CURRENT_BINARY_DIR}/${_py}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_py}
        COMMENT "Syncing ${_py}")
    target_sources(_tender PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${_py})
endforeach()

# Optional: run Python tests if pytest is available.
find_program(PYTEST_EXECUTABLE pytest)
if(PYTEST_EXECUTABLE)
    add_custom_target(python_test
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
            ${PYTEST_EXECUTABLE} test_tender.py -v
        DEPENDS _tender
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running Python tests via pytest")
endif()
