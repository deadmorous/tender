#!/usr/bin/env bash
# Strip outputs and execution counts from all Jupyter notebooks in this directory.
#
# Usage:
#   examples/strip_notebooks.sh        # asks for confirmation
#   examples/strip_notebooks.sh -y     # strips without asking

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t notebooks < <(find "$script_dir" -maxdepth 1 -name "*.ipynb" | sort)

if [ ${#notebooks[@]} -eq 0 ]; then
    echo "No notebooks found in $script_dir"
    exit 0
fi

echo "Notebooks to strip:"
for nb in "${notebooks[@]}"; do
    echo "  $(basename "$nb")"
done

if [ "${1:-}" != "-y" ]; then
    printf '\nStrip all outputs? [y/N] '
    read -r reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

for nb in "${notebooks[@]}"; do
    nbstripout "$nb"
    echo "Stripped: $(basename "$nb")"
done

echo "Done."
