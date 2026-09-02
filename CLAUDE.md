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

Overall accuracy is **not** the target, and neither is macro-F1 — they are
diagnostics. A model that is 90% accurate overall but only clears the bar on 20% of
photos is worse *for this product* than one that is 85% accurate and clears it on
55%. Report every result as **precision above 0.90 confidence** and **% of photos
above 0.90**, in that order, before any other number.

Per-class recall stays a diagnostic: a class that never clears the bar is invisible
in the headline and still a failure.

What that means here: the 0.90 line is the product, not a display detail. The UI
names a species above it and presents the field below it, and the threshold
constant lives in `modal/app.py` (`CONFIDENT_AT`). Moving it changes what the
model promises — measure before touching it.
