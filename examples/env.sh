#!/usr/bin/env bash
# Source this file to put the tender Python package on PYTHONPATH.
#
# The Python bindings are compiled in-place: the .so lands directly in
# python/tender/ inside the repository, so PYTHONPATH only needs to point
# to python/.
#
# Prerequisites — build the Python bindings once:
#
#   cmake -B build-py -G Ninja \
#     -DCMAKE_CXX_COMPILER=g++-14 \
#     -DTENDER_BUILD_PYTHON=ON \
#     -DTENDER_BUILD_TESTS=OFF
#   cmake --build build-py
#
# Usage:
#   source examples/env.sh

_td_script="${BASH_SOURCE[0]:-$0}"
_td_examples_dir="$(cd "$(dirname "$_td_script")" && pwd)"
_td_root="$(dirname "$_td_examples_dir")"
_td_pydir="${_td_root}/python"

if ! ls "${_td_pydir}/tender/_core"*.so 2>/dev/null | grep -q .; then
    echo "ERROR: tender Python module not found in ${_td_pydir}/tender/" >&2
    echo "Build first:" >&2
    echo "  cmake -B build-py -G Ninja -DCMAKE_CXX_COMPILER=g++-14 \\" >&2
    echo "        -DTENDER_BUILD_PYTHON=ON -DTENDER_BUILD_TESTS=OFF" >&2
    echo "  cmake --build build-py" >&2
    unset _td_script _td_examples_dir _td_root _td_pydir
    return 1
fi

export PYTHONPATH="${_td_pydir}${PYTHONPATH:+:${PYTHONPATH}}"
echo "tender ready — PYTHONPATH=${_td_pydir}"
echo "Try: python examples/kronecker_delta.py"

unset _td_script _td_examples_dir _td_root _td_pydir
