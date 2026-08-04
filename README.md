# EggIC Portal

Web portal for the egg-case classifier: identifies which of nine UK shark, ray and
skate species laid an egg case from a photograph, and records what testers make of
each answer.

Training and evaluation live in the sibling `EggIC/` repo. This one only serves what
that repo produces.

```
public/          the frontend — index.html, about.html, assets/
api/             Vercel serverless functions (the browser never sees the Modal key)
modal/           app.py (GPU inference + storage) and inference.py (preprocessing)
local/server.py  development server; stands in for Vercel
```

## How a request flows

```
browser  ──►  /api/predict  ──►  Modal (T4 GPU)  ──►  eggic-submissions volume
              (Vercel fn)        model on the         image + JSON sidecar
              adds the key       eggic-models volume
```

The browser talks only to Vercel, which adds the Modal key server-side. The same
frontend runs locally against `local/server.py`, which mirrors both API routes.

**Local development never stores anything.** `local/server.py` hard-codes
`store: false`; the deployed site sends `store: true` and keeps every submission.

## Deploying

**1. Modal** — repeat after any model or code change:

```bash
modal deploy modal/app.py
```

Requires `EggIC/` checked out as a sibling directory: the image imports the model
definition from `EggIC/eggic-fgvc/src`, so the served network cannot drift from the
weights that produced it. Prints the two endpoint URLs.

**2. Vercel** — set three environment variables in the project:

| Variable | Value |
|---|---|
| `MODAL_PREDICT_URL` | the predict URL from `modal deploy` |
| `MODAL_FEEDBACK_URL` | the feedback URL from `modal deploy` |
| `EGGIC_API_KEY` | must match the `eggic-api-key` Modal secret |

No build step: `public/` is static, `api/` are Node functions.

## Running locally

```bash
cp .env.example .env      # fill in the two URLs and the key
python local/server.py
```

## The model

**v1** = `dinov2_l`: DINOv2 ViT-L/14 at 448px, linear head, temperature 1.0.
84.2% on 101 held-out photographs, with the genus correct on all 101. Four TTA views
(scales 1.0 and 0.85, each mirrored). Above **85%** confidence the interface names a
species; below it, it presents the field.

### Swapping the model

```bash
modal volume put eggic-models best.pt v2/best.pt
modal volume put eggic-models calibration_T1.json v2/calibration_T1.json
MODEL_VERSION=v2 modal deploy modal/app.py
```

Old versions stay on the volume, so rollback is one redeploy. Every stored
submission records the `model_version` that served it, which is what makes two
models comparable on the same real photographs.

## What gets stored

Volume `eggic-submissions`, one image plus one JSON sidecar per submission, filed by
month. There is deliberately **no index file** — two people submitting at the same
moment would overwrite each other's copy of it. The directory listing is the index.

```
2026-08/20260804T153012-a1b2c3d4.jpg     the photograph, as uploaded
2026-08/20260804T153012-a1b2c3d4.json    everything else
```

The stored photograph is what the browser sent: downscaled to 1024px on the long
edge before upload, not the phone original.

```json
{
  "id": "20260804T153012-a1b2c3d4",
  "ts": "2026-08-04T15:30:12.481922+00:00",
  "image_file": "2026-08/20260804T153012-a1b2c3d4.jpg",
  "original_filename": "IMG_4489.jpg",
  "model_version": "v1",
  "prediction": {
    "top": "BlondeRay",
    "top_name": "Blonde Ray",
    "confidence": 0.891234,
    "confident": true,
    "threshold": 0.85,
    "ranked": {
      "BlondeRay": 0.891234,
      "ThornbackRay": 0.019004,
      "CuckooRay": 0.017221,
      "UndulateRay": 0.014882,
      "SmallSpottedCatshark": 0.013115,
      "Nursehound": 0.012440,
      "SmallEyedRay": 0.011902,
      "SpottedRay": 0.010511,
      "FlapperSkate": 0.009691
    }
  },
  "latency_ms": 210,
  "feedback": {
    "correct": true,
    "user_species": "BlondeRay",
    "ts": "2026-08-04T15:30:41.117003+00:00"
  }
}
```

- `ranked` holds all nine probabilities and sums to 1.0.
- `feedback.correct` is `null` until a tester answers, then `true` or `false`.
- `feedback.user_species` is the class key the tester chose. Pressing *Correct*
  mirrors `prediction.top`; pressing *Not correct* leaves it `null` until they pick
  the right species from the list.

### Pulling the data back

```bash
modal volume get eggic-submissions 2026-08 ./inbox
```

Analysis is a loop over the sidecars — filter to `feedback.correct == false` for the
retraining queue. Add a database when that stops being enough, not before.

## Security

The Modal endpoints are public URLs, so both check a shared key (`key` in the request
body) held in the `eggic-api-key` Modal secret and in the Vercel environment. That
stops casual traffic; it is not a serious access-control system, which is
proportionate for a demo that will be torn down.

Cost control matters more than access control here: the GPU scales to zero after a
60-second idle window, so an abandoned tab stops costing anything a minute later.

## Frontend behaviour

Single screen, no page scroll: photograph on the left, verdict and top three species
on the right, the remaining six behind a disclosure. Light by default, with a theme
toggle persisted in localStorage.

**Batches.** Up to 10 photographs at a time, identified one after another with
results appearing as they land, so review starts on the first while the rest are
still running. A filmstrip jumps between them; `←`/`→` also work. The file picker
cannot cap selection, so the cap is enforced afterwards and a banner names the files
that were left out.

**Feedback.** `Y` / `N` or the two buttons. *Not correct* opens a fixed alphabetical
list of all nine species — "Choose the correct species" — and both answers post to
`/api/feedback` against the stored submission.

**Upload size.** The browser downscales to 1024px on the long edge before upload; the
model works at 448px, so sending a 12MP original only makes the user wait on their own
bandwidth. This measurably shifts confidences by about a percentage point (90.0% →
89.1% on one benchmark photograph), so it is a real change to the model's input and
not purely a transport optimisation. Worth measuring across the full benchmark.
