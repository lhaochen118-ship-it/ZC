# Showcase repo figures

Sources of the figures in the public showcase repo `nhatlong113/ZAIC-2026` (`assets/*.png`, light + dark variants; v2, 12/11: public 0.994 = container run of P).

```bash
python3 gen.py out                               # writes out/<figure>-<theme>.html (one SVG each)
NODE_PATH=$(npm root -g) node render.js out png  # transparent 2x PNGs + preview_*.png on the GitHub page colour
```

Copy `png/{pipeline,example,results}-{light,dark}.png` into the showcase repo's `assets/`. Keep the public figures free of
the model name, training recipe and the content of the verification checks.
