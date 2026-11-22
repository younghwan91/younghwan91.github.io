<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="younghwan91.github.io" src="docs/images/brand/wordmark-light.svg" height="56">
</picture>

[![Live Site](https://img.shields.io/badge/Live-younghwan91.github.io-008080)](https://younghwan91.github.io/)

**Personal résumé homepage styled as a late-90s desktop — where each section of the CV opens as an interactive program window.**

Live at **[younghwan91.github.io](https://younghwan91.github.io/)**.

---

## Highlights

- **Retro Desktop Metaphor:** Classic `#008080` teal desktop, `#c0c0c0` beveled window chrome, and `Pixelify Sans` / `VT323` / `Nanum Gothic Coding` typography.
- **True-Scale Career Timeline:** `timeline.exe` plots roles across employers with peer-reviewed papers and patents marked at true time scale.
- **Zero-Build Static Site:** Self-contained [`index.html`](index.html) served directly by GitHub Pages with [`.nojekyll`](.nojekyll).
- **Reproducible Figure Assets:** Research and perception figures generated via [`tools/make_figures.py`](tools/make_figures.py).

## Local Preview

```bash
python3 -m http.server 8000
# Open http://localhost:8000
```

To regenerate the figure PNGs (`r_goals.png`, `r_sensors.png`, `r_subdim.png`):

```bash
python3 tools/make_figures.py
```

## Repository Layout

| Path | Purpose |
|------|---------|
| [`index.html`](index.html) | Single-page desktop UI, styles, and interactive window logic |
| [`tools/make_figures.py`](tools/make_figures.py) | Script that generates research/perception figure PNGs |
| [`.nojekyll`](.nojekyll) | Disables Jekyll processing on GitHub Pages |
| [`robots.txt`](robots.txt) | Crawler configuration |
| `me.jpg`, `me_px.png`, `w_*.jpg`, `r_*.png`, `logo_*` | Portrait, work, figure, and employer logo assets |
