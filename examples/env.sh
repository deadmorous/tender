#!/usr/bin/env bash
# Source this file to put the tender Python package on PYTHONPATH.
#
# Prerequisites — build the Python bindings once:
#
#   cmake -B build -G Ninja \
#     -DCMAKE_CXX_COMPILER=g++-14 \
#     -DTENDER_BUILD_PYTHON=ON \
#     -DTENDER_BUILD_TESTS=OFF
#   cmake --build build
#
# Usage:
#   source examples/env.sh           # uses 'build' as the build directory
#   source examples/env.sh build-py  # use a different build directory

_td_script="${BASH_SOURCE[0]:-$0}"
_td_examples_dir="$(cd "$(dirname "$_td_script")" && pwd)"
_td_root="$(dirname "$_td_examples_dir")"
_td_build="${1:-build}"
_td_pydir="${_td_root}/${_td_build}/python"

if [ ! -f "${_td_pydir}/tender/_tender"*.so 2>/dev/null ] && \
   [ ! -f "${_td_pydir}/tender/_tender"*.pyd 2>/dev/null ]; then
    echo "ERROR: tender Python module not found in ${_td_pydir}/tender/" >&2
    echo "Build first:" >&2
    echo "  cmake -B ${_td_build} -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DTENDER_BUILD_PYTHON=ON -DTENDER_BUILD_TESTS=OFF" >&2
    echo "  cmake --build ${_td_build}" >&2
    unset _td_script _td_examples_dir _td_root _td_build _td_pydir
    return 1
fi

export PYTHONPATH="${_td_pydir}${PYTHONPATH:+:${PYTHONPATH}}"
echo "tender ready — PYTHONPATH=${_td_pydir}"
echo "Try: python examples/pvw_continuum.py"

unset _td_script _td_examples_dir _td_root _td_build _td_pydir
