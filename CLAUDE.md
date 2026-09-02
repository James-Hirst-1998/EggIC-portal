# CLAUDE.md

The portal that serves the egg-case classifier. Read `README.md` for how it is
built and deployed; this file is what the work is *for*.

Training and evaluation live in the sibling `EggIC/` repo (`eggic-fgvc/`), which
carries the same goal statement. Keep the two copies in step.

## The goal — the only two numbers that matter

1. **Everything the model calls at ≥85% confidence must be right.** A wrong answer
   above that line is the failure this whole project exists to avoid. Errors *below*
   the line are acceptable — that is what "not sure" is for.
2. **Get at least 50% of photos above that line.** Half of real submissions answered
   automatically, every one of them correct, is the golden goal.

The bar is **0.85, not 0.90**, because label smoothing caps this model's confidence
near 0.911 — a 0.90 cut sits in the top 1.5% of a scale the recipe squashed. 0.85 is
the same intent measured on the scale the model actually has, and it is already the
portal's `CONFIDENT_AT`.

Overall accuracy is **not** the target, and neither is macro-F1 — they are
diagnostics. A model that is 90% accurate overall but only clears the bar on 20% of
photos is worse *for this product* than one that is 85% accurate and clears it on
55%. Report every result as **precision above 0.85 confidence** and **% of photos
above 0.85**, in that order, before any other number.

Per-class recall stays a diagnostic: a class that never clears the bar is invisible
in the headline and still a failure.

**Where it stands** *(2026-09-02, 224 held-out photos)* — v1+DINOv3 averaged: **100 of
224 above 0.85, none of them wrong** (rule 1 met, rule 2 twelve photos short). DINOv3
alone: 90 above, none wrong. **v1 alone fails rule 1** — 6 wrong answers above 0.85.

What that means here: the 0.90 line is the product, not a display detail. The UI
names a species above it and presents the field below it, and the threshold
constant lives in `modal/app.py` (`CONFIDENT_AT`). Moving it changes what the
model promises — measure before touching it.
