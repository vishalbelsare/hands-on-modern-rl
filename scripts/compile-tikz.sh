#!/bin/bash
# Compile all TikZ files to SVG
# Usage: bash scripts/compile-tikz.sh
set -e

export PATH="/Library/TeX/texbin:$PATH"

TIKZ_DIR="$(cd "$(dirname "$0")/tikz" && pwd)"
DOCS_DIR="$(cd "$(dirname "$0")/../docs/appendix_math" && pwd)"
OUT_DIR="$DOCS_DIR/images"
mkdir -p "$OUT_DIR"

echo "TikZ dir: $TIKZ_DIR"
echo "Output dir: $OUT_DIR"

for tex in "$TIKZ_DIR"/*.tex; do
    name=$(basename "$tex" .tex)
    echo -n "  $name ... "

    workdir=$(mktemp -d)
    cp "$tex" "$workdir/"
    cd "$workdir"

    pdflatex -interaction=nonstopmode "$name.tex" > /dev/null 2>&1
    pdf2svg "$name.pdf" "$OUT_DIR/${name}.svg" 2>/dev/null

    rm -rf "$workdir"
    echo "OK"
done

echo "Done! All TikZ files compiled to $OUT_DIR/"
