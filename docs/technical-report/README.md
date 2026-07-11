# Technical Report

System architecture and design report for the mesh agent framework.

## Building

Requires TeX Live (`pdflatex`):

```bash
# Debian/Ubuntu
sudo apt-get install texlive-latex-recommended texlive-latex-extra

# Build PDF (runs pdflatex twice for table of contents)
make pdf

# Clean build artifacts
make clean
```

The pre-built `mesh_technical_report.pdf` is included so a TeX installation
is not required to read the report.
