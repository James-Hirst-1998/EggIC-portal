# CLAUDE.md

The portal that serves the egg-case classifier. Read `README.md` for how it is
built and deployed; this file is what the work is *for*.

Training and evaluation live in the sibling `EggIC/` repo (`eggic-fgvc/`), which
carries the same goal statement. Keep the two copies in step.

## The goal — the only two numbers that matter

1. **Everything the model answers must be right.** The accept line is **87.5%** on the shown scale. A wrong answer
   above that line is the failure this whole project exists to avoid. Errors *below*
   the line are acceptable — that is what "not sure" is for.
2. **Get at least 50% of photos above that line.** Half of real submissions answered
   automatically, every one of them correct, is the golden goal.

**90% means the shown scale**, not raw softmax. Label smoothing caps raw confidence
near 0.911, so the raw number could never express certainty. Every arm is displayed
through one shared `display_temperature = 0.80` (`softmax(log p / T)`, renormalised):
it moves tail mass to the leader, leaves the species and its ranking untouched, caps
the dial at **96.8%** and puts the averaged arm's worst observed mistake at **83.5%**
— a 4-point buffer under the 87.5% accept line. Raw 0.72 shows as 84%. One shared T, not
a per-arm fit, so the arms stay comparable and a weak arm is not flattered.

Overall accuracy is **not** the target, and neither is macro-F1 — they are
diagnostics. Report every result as **precision above the line** and **% of photos
above the line**, in that order, before any other number. Per-class recall stays a
diagnostic: a class that never clears the bar is invisible in the headline and still
a failure.

**Where it stands** *(2026-09-02, 224 held-out photos, shown scale, accept at 0.875)* —
v1+DINOv3 averaged: **133 of 224 (59%), none wrong** — both rules met. On the same
scale DINOv3 alone answers 135 with **5 wrong** and v1 alone answers 177 with **15
wrong**, so serving the average is a correctness fix, not a preference. Caveats: the
scale and the line were fitted on this same sample (95% floor ≈97%, not 100%), and 5
of the 19 untaught-species photos still clear 0.90.

What that means here: the 0.90 line is the product, not a display detail. The UI
names a species above it and presents the field below it, and the threshold
constant lives in `modal/app.py` (`CONFIDENT_AT`). Moving it changes what the
model promises — measure before touching it.
