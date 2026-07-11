# Mesh Technical Report — Build
#
# Prerequisites:
#   sudo apt-get install texlive-latex-recommended texlive-latex-extra
#   (or equivalent TeX Live distribution)
#
# Usage:
#   make pdf      # Build the PDF (runs pdflatex twice for TOC)
#   make clean    # Remove build artifacts

TEX = mesh_technical_report.tex
PDF = $(TEX:.tex=.pdf)

.PHONY: pdf clean

pdf: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode $(TEX) || true
	pdflatex -interaction=nonstopmode $(TEX) || true
	@test -f $(PDF) && echo "Built $(PDF)" || (echo "ERROR: PDF not produced" && exit 1)

clean:
	rm -f *.aux *.log *.out *.toc
