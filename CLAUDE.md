# CLAUDE.md

The portal that serves the egg-case classifier. Read `README.md` for how it is
built and deployed; this file is what the work is *for*.

Training and evaluation live in the sibling `EggIC/` repo (`eggic-fgvc/`), which
carries the same goal statement. Keep the two copies in step.

## The goal — the only two numbers that matter

1. **Everything the model calls at ≥90% confidence must be right.** A wrong answer
   above that line is the failure this whole project exists to avoid. Errors *below*
   the line are acceptable — that is what "not sure" is for.
2. **Get at least 50% of photos above that line.** Half of real submissions answered
   automatically, every one of them correct, is the golden goal.

**90% means the corrected display scale**, not raw softmax. Label smoothing caps raw
confidence near 0.911, so the raw number could never express certainty; each arm now
carries a `display_temperature` fitted on the 224 held-out photos so that its highest
*observed wrong answer* lands just under 0.90. On that scale "90% or above" means
"no error was observed here", it tops out at 98.6% rather than 100%, and it is better
calibrated than the raw number (ensemble ECE 0.143 → 0.048). Argmax and ranking are
untouched. The equivalent raw cut for the averaged arm is **0.72**.

Overall accuracy is **not** the target, and neither is macro-F1 — they are
diagnostics. Report every result as **precision above the line** and **% of photos
above the line**, in that order, before any other number.

Per-class recall stays a diagnostic: a class that never clears the bar is invisible
in the headline and still a failure.

**Where it stands** *(2026-09-02, 224 held-out photos, displayed scale)* — v1+DINOv3
averaged: **143 of 224 above 0.90 (64%), none wrong** — both rules met on this sample.
DINOv3 alone: 93 above, none wrong. **v1 alone cannot meet rule 1 at any line** — its
worst mistake is made at 0.906 raw, above any threshold you could set. Caveats: the
cut was fitted on this same sample (95% floor is ~97%, not 100%), and 6 of the 19
untaught-species photos still sit above the line.

What that means here: the 0.90 line is the product, not a display detail. The UI
names a species above it and presents the field below it, and the threshold
constant lives in `modal/app.py` (`CONFIDENT_AT`). Moving it changes what the
model promises — measure before touching it.
