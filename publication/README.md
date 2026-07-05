# Manuscript — Applied Energy submission

Files:
- `elsarticle-template-num.tex` — **submission master** (Elsevier `elsarticle`, numbered refs).
- `references.bib` — 33 references (BibTeX).
- `paper.md` — Markdown mirror for co-author editing / `.docx` export. Keep in sync with the `.tex`.
- `img/turbine.JPG` — test-rig photo (Fig. 1).
- `CFDBOR4.pdf` — colleague's original draft (superseded by the `.tex`).

Figures are pulled from `../result/` (study output) and `img/`. The figure-generating
scripts live at the repo root, next to the study code. Regenerate if the results change:

```bash
uv run python bayesian_reduction.py   # study + learning curves, bars, placement, contrast
uv run python make_figures.py         # all-methods 3D/2D/error + uncertainty panels
```

## Build the PDF

```bash
cd publication
pdflatex elsarticle-template-num
bibtex   elsarticle-template-num
pdflatex elsarticle-template-num
pdflatex elsarticle-template-num
```

(`latexmk -pdf elsarticle-template-num.tex` also works outside a privileged shell.)
Needs the `elsarticle` class + `elsarticle-num.bst` (ships with TeX Live / MiKTeX).

## Export the editable Word file

```bash
pandoc paper.md -o paper.docx
```
