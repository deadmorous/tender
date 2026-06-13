find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

nanobind_add_module(_core NB_STATIC _core.cpp)
target_link_libraries(_core PRIVATE tender::tender)
target_compile_options(_core PRIVATE -Wall -Wextra -Werror)

# Output the .so next to the Python package files for in-tree development.
# Set PYTHONPATH to python/ to use: `PYTHONPATH=python python -c "import tender"`
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tender"
)
