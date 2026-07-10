# Run examples: generate LaTeX and compile to PDF.
#
# Usage:
#   make                       # build all examples
#   make kronecker_delta       # build one example
#   make clean                 # remove the out/ directory
#
# Prerequisites:
#   1. Build the tender Python bindings:
#        cmake -B build-py -G Ninja -DCMAKE_CXX_COMPILER=g++-14 \
#              -DTENDER_BUILD_PYTHON=ON -DTENDER_BUILD_TESTS=OFF
#        cmake --build build-py
#   2. Have pdflatex installed.
#
# The .so lives in python/tender/ (built in-place), so no build-dir path
# is needed here — PYTHONPATH just points to python/.

ROOT     := ..
PYTHON   ?= python3
PDFLATEX ?= pdflatex
OUTDIR   := out

PYTHONPATH_VAR := PYTHONPATH=$(ROOT)/python

EXAMPLES := kronecker_delta delta_trace eps_delta basis_dot_product cross_identity \
            basis_aware_coordinates curvilinear_operators cyl_equilibrium \
            strain_compatibility navier_lame

.PHONY: all clean

all: $(EXAMPLES)

$(OUTDIR):
	mkdir -p $(OUTDIR)

%: %.py | $(OUTDIR)
	$(PYTHONPATH_VAR) $(PYTHON) $<
	$(PDFLATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(OUTDIR)/$@.tex

clean:
	rm -rf $(OUTDIR)
