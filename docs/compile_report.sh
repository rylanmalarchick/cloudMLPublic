#!/bin/bash

# Script to compile LaTeX project status report to PDF
# Usage: ./compile_report.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="$SCRIPT_DIR/project_status_report.tex"
OUTPUT_DIR="$SCRIPT_DIR"

echo "=================================================="
echo "Compiling Project Status Report"
echo "=================================================="

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo "Please install LaTeX (e.g., texlive-latex-base on Ubuntu/Debian)"
    echo ""
    echo "Installation commands:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo "  macOS: brew install --cask mactex"
    echo "  Fedora: sudo dnf install texlive-scheme-basic"
    exit 1
fi

cd "$OUTPUT_DIR"

echo "Compiling LaTeX document..."
echo ""

# First pass
echo "[1/3] First compilation pass..."
pdflatex -interaction=nonstopmode "$(basename "$TEX_FILE")" > /dev/null 2>&1 || {
    echo "ERROR: First compilation pass failed!"
    echo "Running with output for debugging:"
    pdflatex -interaction=nonstopmode "$(basename "$TEX_FILE")"
    exit 1
}

# Second pass (for table of contents)
echo "[2/3] Second compilation pass (table of contents)..."
pdflatex -interaction=nonstopmode "$(basename "$TEX_FILE")" > /dev/null 2>&1 || {
    echo "WARNING: Second pass had errors, but continuing..."
}

# Third pass (for cross-references)
echo "[3/3] Third compilation pass (cross-references)..."
pdflatex -interaction=nonstopmode "$(basename "$TEX_FILE")" > /dev/null 2>&1 || {
    echo "WARNING: Third pass had errors, but continuing..."
}

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

OUTPUT_PDF="${TEX_FILE%.tex}.pdf"

if [ -f "$OUTPUT_PDF" ]; then
    echo ""
    echo "=================================================="
    echo " SUCCESS!"
    echo "=================================================="
    echo "PDF generated: $OUTPUT_PDF"
    echo ""
    echo "File size: $(du -h "$OUTPUT_PDF" | cut -f1)"
    echo "Pages: $(pdfinfo "$OUTPUT_PDF" 2>/dev/null | grep Pages | awk '{print $2}' || echo 'unknown')"
    echo ""
    echo "To view the PDF:"
    echo "  evince $OUTPUT_PDF"
    echo "  # or"
    echo "  xdg-open $OUTPUT_PDF"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "ERROR: PDF was not generated!"
    echo "=================================================="
    echo "Check the LaTeX log for errors."
    exit 1
fi
