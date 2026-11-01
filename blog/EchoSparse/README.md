# EchoSparse blog

Author: Shitong Shao (yuqs). Published at https://w2genai-lab.github.io/blog/EchoSparse/ .

- `index.html` (中文) / `en.html` (English): published reading pages; they also open directly in a browser. They embed the original Canvas animation (`assets/echosparse-canvas-animation.html`, unmodified) with a play/replay button, and keep the GIF for reposting.
- `echosparse_zh.md` / `echosparse_en.md`: publication drafts (figure paths are relative, PNG for Zhihu upload).
- `assets/`: SVG + 2x PNG figures, GIF, original animation, local Inter (latin) and KaTeX assets.
- `sources/`: editable `.drawio` diagrams and build scripts. `data/`, `evidence/`: plotted observations and provenance.

Rebuild (needs Python 3, Node with `markdown-it`, `markdown-it-texmath`, `katex`, `playwright`+chromium):

    python3 sources/make_figures.py && python3 sources/make_plots.py
    node sources/export_png.js assets          # SVG -> 2x PNG
    node sources/build_pages.js                # md -> index.html / en.html
