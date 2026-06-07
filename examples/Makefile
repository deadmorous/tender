# Run examples: generate LaTeX, compile to PDF, and open the result.
#
# Usage:
#   make                      # build all examples
#   make pvw_continuum        # build one example
#   make VIEWER=evince        # use a specific PDF viewer
#   make BUILD=build-py       # use a different CMake build directory
#   make clean                # remove the out/ directory
#
# Prerequisites:
#   1. Build the tender Python bindings:
#        cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=g++-14 \
#              -DTENDER_BUILD_PYTHON=ON -DTENDER_BUILD_TESTS=OFF
#        cmake --build build
#   2. Have pdflatex and a PDF viewer installed.

BUILD    ?= ../build
PYTHON   ?= python3
PDFLATEX ?= pdflatex
VIEWER   ?= xdg-open
OUTDIR   := out

PYTHONPATH_VAR := PYTHONPATH=$(BUILD)/python

EXAMPLES := pvw_continuum bac_cab bac_cab_search dot_commutativity

.PHONY: all clean

all: $(EXAMPLES)

$(OUTDIR):
	mkdir -p $(OUTDIR)

%: %.py | $(OUTDIR)
	$(PYTHONPATH_VAR) $(PYTHON) $<
	$(PDFLATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(OUTDIR)/$@.tex
	@$(VIEWER) $(OUTDIR)/$@.pdf 2>/dev/null &

clean:
	rm -rf $(OUTDIR)
