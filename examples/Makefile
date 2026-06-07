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

EXAMPLES := pvw_continuum bac_cab bac_cab_search

.PHONY: all $(EXAMPLES) clean

all: $(EXAMPLES)

$(OUTDIR):
	mkdir -p $(OUTDIR)

pvw_continuum: | $(OUTDIR)
	$(PYTHONPATH_VAR) $(PYTHON) pvw_continuum.py
	$(PDFLATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(OUTDIR)/pvw_continuum.tex
	@$(VIEWER) $(OUTDIR)/pvw_continuum.pdf 2>/dev/null &

bac_cab: | $(OUTDIR)
	$(PYTHONPATH_VAR) $(PYTHON) bac_cab.py
	$(PDFLATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(OUTDIR)/bac_cab.tex
	@$(VIEWER) $(OUTDIR)/bac_cab.pdf 2>/dev/null &

bac_cab_search: | $(OUTDIR)
	$(PYTHONPATH_VAR) $(PYTHON) bac_cab_search.py
	$(PDFLATEX) -interaction=nonstopmode -output-directory=$(OUTDIR) $(OUTDIR)/bac_cab_search.tex
	@$(VIEWER) $(OUTDIR)/bac_cab_search.pdf 2>/dev/null &

clean:
	rm -rf $(OUTDIR)
