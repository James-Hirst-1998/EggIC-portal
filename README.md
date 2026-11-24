# EggIC Portal

Web portal for the egg-case classifier: identifies which of nine UK shark, ray and
skate species laid an egg case from a photograph, and records what testers make of
each answer.

Training and evaluation live in the sibling `EggIC/` repo. This one only serves what
that repo produces.

```
public/          the frontend — index.html, about.html, test.html, assets/
public/js/       ES modules for test.html; Supabase calls only in services/
api/             Vercel serverless functions (the browser never sees the Modal key)
modal/           app.py (GPU inference + storage) and inference.py (preprocessing)
supabase/        migrations/ — the database, buckets and access rules
scripts/         one-off jobs against Supabase: import, staff-test setup, testers
local/server.py  development server; stands in for Vercel
```

## How a request flows

```
browser  ──►  /api/predict  ──►  Modal (L4 GPU)  ──►  Supabase
              (Vercel fn)        model on the         photo to a bucket,
              adds the key       eggic-models volume  answer to a table
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

Once only, before the first deploy that stores to Supabase:

```bash
modal secret create eggic-supabase SUPABASE_URL=https://<project-ref>.supabase.co SUPABASE_SECRET_KEY=sb_secret_...
```

Requires `EggIC/` checked out as a sibling directory: the image imports the model
definition from `EggIC/eggic-fgvc/src`, so the served network cannot drift from the
weights that produced it. Prints the two endpoint URLs.

**2. Vercel** — set three environment variables in the project:

| Variable | Value |
|---|---|
| `MODAL_PREDICT_URL` | the predict URL from `modal deploy` |
| `MODAL_FEEDBACK_URL` | the feedback URL from `modal deploy` — note this is `--eggic-feedback`, *not* `--eggic-model-feedback` |
| `EGGIC_API_KEY` | must match the `eggic-api-key` Modal secret |

No build step: `public/` is static, `api/` are Node functions.

Nothing else needs setting. The PostHog project token in `public/analytics.js` is a
public write key and belongs in the page; `vercel.json` rewrites `/ingest/*` onto
`eu.i.posthog.com` so the calls are first-party and survive tracker blockers.

## Photo formats

JPEG, PNG, WebP and HEIC. Video is rejected with a message naming the file.

HEIC is what an iPhone shoots by default. Safari and every iOS browser decode it
natively and take the same path as a JPEG. Chrome, Firefox and Edge cannot, so on
those `public/vendor/heic-to.js` (libheif compiled to WebAssembly, 3MB, vendored so
there is no CDN in the critical path) is fetched the first time a HEIC appears and
decodes it in a worker. Budget several seconds per 12-megapixel photo on that path —
conversion is therefore run one photo ahead of identification, so it overlaps the
previous photo's round trip rather than following it.

## Running locally

```bash
cp .env.example .env      # fill in the two URLs and the key
python local/server.py
```

`/test` works locally too, but it talks straight to Supabase: answers given there
are real.

## The model

Two arms are loaded at once (`MODEL_VERSIONS`, default `s2_dinov2,s2_dinov3`) and their
raw probabilities are averaged. Four TTA views each (scales 1.0 and 0.85, mirrored). Eleven
classes: the nine species, `RareSpecies` (Starry, Blue, White Skate, Blackmouth Catshark,
shown as "send to an expert") and `NoEggCase`. At or above `CONFIDENT_AT` (**0.95** on the
shown scale) the interface names an answer; below it, it presents the field.

| arm | backbone | answered at 0.95, of 224 held-out photos | wrong |
|---|---|---|---|
| `s2_dinov2` | DINOv2 ViT-L/14 @448 | 207 | 3 |
| `s2_dinov3` | DINOv3 ViT-L/16 @448 | 183 | 1 |
| both, averaged | — | **190 (85%)** | **0** |

Measured 2026-11-17 on the 101 benchmark + 123 hand-verified field photos, none of them in
training. The averaged arm's most confident mistake is 93.7%, so 0.95 is the lowest clean
line; it was chosen on this same sample. See `CLAUDE.md` for the goal those numbers are
judged against.

### Swapping the model

```bash
modal volume put eggic-models best.pt v2/best.pt
modal volume put eggic-models calibration_T1.json v2/calibration_T1.json
MODEL_VERSIONS=s2_dinov2,s2_dinov3,v2 modal deploy modal/app.py
```

`final_ema.pt` is preferred over `best.pt` when a run wrote both. The model code comes
from the sibling `EggIC/` checkout; when that lags `main`, point at a checkout of `main` —
`EGGIC_SRC=../EggIC-<branch>/eggic-fgvc/src modal deploy modal/app.py`.
The deploy fails loudly rather than serving a checkout that cannot build DINOv3.

### Segmentation

Every photo is segmented before it is named (EggIC `multispecies/segment.py`: SAM
vit_h proposes masks, a keep head keeps whole egg cases). When it finds two or more
cleanly, each is cropped and named on its own; more than eight (`MAX_CASES` in
`modal/app.py`) is `too_many` and none is named; otherwise the whole photo is named as one
case, outlined if the segmenter found exactly one. When the segmenter is `unsure` of the
count, that case is never `confident`, however high it scores. `"segment": false` skips it. The
head and the crop fill are copied, so a retrain in EggIC changes nothing until recopied:

```bash
modal volume put eggic-models keep_head_ac.npz segment/keep_head.npz
modal volume put eggic-models meta.json segment/meta.json   # {"fill_target": 0.7011}
```

Old versions stay on the volume, so rollback is one redeploy. Every stored
submission records the `model_version` that served it, which is what makes two
models comparable on the same real photographs.

## What the API returns

One photo in, a list of egg cases out. One case and five come back the same way; none is
an empty list.

```json
{
  "id": "20261119T141007-c7d4a330",   // the stored submission; null when not stored
  "model_version": "ensemble",        // the arm that answered
  "threshold": 0.95,                  // the accept line: at or above it, a case is "confident"
  "latency_ms": 2140,
  "segmenter": {"status": "ok", "seconds": 1.8, "max_cases": 8},  // ok | no_egg_case | too_many (cases empty) | unsure | skipped
  "count": 2,
  "no_egg_case": null,                // when cases is empty: how sure the model was there is none
  "cases": [
    {
      "n": 1,                                   // 1, 2, … top to bottom
      "box": [0.10, 0.22, 0.34, 0.58],          // x0, y0, x1, y1 as fractions of the photo
      "outline": [[[0.12, 0.23], [0.30, 0.25], …]],  // the mask's edge, same fractions
      "mask_path": "2026-11/20261119T141007-c7d4a330/1.png",  // in the submissions bucket
      "top": "ThornbackRay", "top_name": "Thornback Ray",
      "confidence": 0.97, "confident": true,
      "ranked": [{"key": "ThornbackRay", "name": "Thornback Ray", "latin": "Raja clavata", "prob": 0.97}, …],
      "by_model": {"s2_dinov2": {"top": "ThornbackRay", "top_name": "Thornback Ray", "confidence": 0.96}, …}
    }
  ]
}
```

`ranked` holds all eleven classes on the shown scale, best first. A case named from the
whole photo has `box`, `outline` and `mask_path` null. A case the model is sure is not an
egg case is left out. Fractions, not pixels, so they fit the photo at any size.

## What gets stored

Everything is in one Supabase project; `supabase/migrations/` is the whole schema.

| | holds | readable by |
|---|---|---|
| bucket `submissions` | every photo the live site identified, `YYYY-MM/<id>.jpg`, and each case's mask, `YYYY-MM/<id>/<n>.png` | secret key only |
| table `submissions` | one row per photo: file, model, accept line, segmenter verdict, how many cases | secret key only |
| table `cases` | one row per egg case: box, outline, mask file, answer, every class's score, each arm's answer, the tester's verdict | secret key only |
| bucket `test-images` | the staff-test photos, under random names | logged-in testers |
| table `test_images` | the true species of each test photo, and what the model said | secret key only |
| table `test_answers` | one row per tester per photo: the species they chose and an optional note | each tester, own rows only |

The stored photograph is what the browser sent: downscaled to 1024px on the long
edge, not the phone original. In `cases`, `correct` is null until a tester answers
that case; `user_species` is the class they chose. `threshold` on `submissions` is the
accept line in force when the photo was served. Photos from before 2026-11-19 were named
whole: each has one case with no outline, and keeps its old answer columns on
`submissions` too. Submissions from before Supabase were copied in from the old Modal
volume with `scripts/import_volume.py`.

### Pulling the data back

In the Supabase SQL editor:

```sql
-- the retraining queue: every case a tester marked wrong, with its photo
select s.image_path, c.* from cases c join submissions s on s.id = c.submission_id
where c.correct = false order by s.created_at desc;

-- staff on the test, scored like the model: right when sure, then how often sure
select u.email, count(*) as answered,
       round(avg((a.species = t.true_species)::int) filter (where a.species <> 'not_sure'), 3) as right_when_sure,
       round(avg((a.species <> 'not_sure')::int), 3) as sure
from test_answers a join test_images t using (image_path) join auth.users u on u.id = a.user_id
group by u.email;

-- every note, beside the true species
select u.email, t.true_species, a.species, a.note
from test_answers a join test_images t using (image_path) join auth.users u on u.id = a.user_id
where a.note is not null;

-- the model on the same photos: precision above the line, then share above it
select round(avg((model_top = true_species)::int) filter (where model_confidence >= 0.875), 3) as right_above_line,
       round(avg((model_confidence >= 0.875)::int), 3) as above_line
from test_images;
```

### Why feedback is a separate function

`feedback` is a plain CPU function, not a method on `Model`. As a method it inherited
`@modal.enter()`, so recording a one-word answer woke a T4 and loaded 1.2 GB of
weights first — tens of seconds, and billed as GPU time. It now runs in a small
container that holds the Supabase secret and nothing else.

## Staff test

`/test` — Shark Trust staff log in, then name the species in 150 photos, one at a
time: nine buttons plus *Not sure*, keys `1`–`9` and `0`, and an optional note saved
with the answer. Scroll or pinch zooms the photo. Each tester gets their own fixed
shuffle, can skip photos (*Unanswered only* steps through the ones left) and can leave
and come back. An answer moves straight on and saves in the background; a failed save
leaves the photo unanswered and says so. Not linked from the site, and `noindex`.

PostHog sees the page like any other, with each tester identified by their login
email, plus `test_started`, `test_answered` (species, seconds on the photo, zoom,
note, changed), `test_moved` (next/back, key or button, from an unanswered photo),
`test_filter`, `test_save_failed`, `test_finished` and `test_login_failed`. The
answers themselves are in `test_answers`; PostHog is for how people get there.

```bash
python scripts/pick_test_set.py > test_set.csv     # which photos, with true species
python scripts/load_test_set.py test_set.csv       # make the set match: add, upload, remove
python scripts/score_test_set.py                   # the live model's answer (GPU)
python scripts/add_tester.py name@sharktrust.org   # prints their password; no email sent
```

The set leans on the species the model finds hardest (Thornback and Blonde most,
Cuckoo and Flapper least), and within each, on photos the live model got wrong or
answered below the line, plus damaged cases. Photos are held out only: the benchmark,
the Shark Trust export's test tier, and portal submissions. A portal photo's file is
named after the model's answer, so its true species comes from the tester's feedback
in `submissions`, and one with no feedback is left out. `EXCLUDE` in
`pick_test_set.py` lists photos ruled out by eye (two cases, species written in shot).

The export is private (people's faces): its photos only ever go to the `test-images`
bucket, read through 6-hour signed links by tester accounts.

## Security

Logins are Supabase email + password, created by `scripts/add_tester.py` with
`app_metadata.tester` set. Only the secret key can set that flag, and every policy
checks it, so an account made any other way sees nothing.

The Modal endpoints are public URLs, so both check a shared key (`key` in the request
body) held in the `eggic-api-key` Modal secret and in the Vercel environment. That
stops casual traffic; it is not a serious access-control system, which is
proportionate for a demo that will be torn down.

Cost control matters more than access control here: the GPU scales to zero after a
60-second idle window, so an abandoned tab stops costing anything a minute later.

## Frontend behaviour

Single screen, no page scroll (stacked on a phone): photograph on the left with each egg
case outlined, one card per case on the right, top three species shown and the rest behind
a disclosure. Light by default, with a theme toggle persisted in localStorage.

**Batches.** Up to 20 photographs at a time, identified one after another with
results appearing as they land, so review starts on the first while the rest are
still running. A filmstrip jumps between them; `←`/`→` also work. The file picker
cannot cap selection, so the cap is enforced afterwards and a banner names the files
that were left out.

**Egg cases.** One card per case, the same for one as for five. A card opens to its
scores and its own feedback; each case's mask can be hidden on its own or all at once,
and clicking a mask on the photo opens its card.

**Feedback.** `Y` / `N` or the two buttons, on the open case. *Not correct* opens a fixed
alphabetical list of all nine species — "Choose the correct species" — and both answers
post to `/api/feedback` against that case. Then the next unanswered case, or photo.

**Upload size.** The browser downscales to 1024px on the long edge before upload; the
model works at 448px, so sending a 12MP original only makes the user wait on their own
bandwidth. This measurably shifts confidences by about a percentage point (90.0% →
89.1% on one benchmark photograph), so it is a real change to the model's input and
not purely a transport optimisation. Worth measuring across the full benchmark.
