find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
nanobind_add_module(_tender NB_STATIC _tender.cpp)

target_link_libraries(_tender PRIVATE tender::tender)

# Put the compiled extension inside the tender/ package directory so that
# PYTHONPATH=<build>/python makes "import tender" work.
set_target_properties(_tender PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tender)

# Copy the Python package source into the build tree.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/tender/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/tender/__init__.py
    COPYONLY)

# Copy the test file to the build tree so pytest runs from there,
# avoiding sys.path conflicts with the source python/ directory.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tender.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_tender.py
    COPYONLY)

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
