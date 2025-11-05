#!/bin/bash
# Compile LaTeX paper
# Usage: ./compile_paper.sh

echo "Compiling paper.tex..."

# Run pdflatex twice for references
pdflatex paper.tex
pdflatex paper.tex

# Clean up auxiliary files
rm -f paper.aux paper.log paper.out

echo "Done! Output: paper.pdf"
