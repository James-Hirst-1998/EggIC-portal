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

Two arms are loaded at once (`MODEL_VERSIONS`, default `v1,dinov3`) and the picker on
the front page chooses between them or averages both. Four TTA views each (scales 1.0
and 0.85, mirrored). Above **85%** confidence the interface names a species; below it,
it presents the field.

| arm | backbone | answers with zero errors, of 224 held-out photos |
|---|---|---|
| `v1` | DINOv2 ViT-L/14 @448 | 40 (18%) |
| `dinov3` | DINOv3 ViT-L/16 @448 | 97 (43%) |
| both, averaged | — | **145 (65%)** |

"Zero errors" is the largest set of most-confident photos containing no wrong answer,
measured on the 101 benchmark + 123 hand-verified field photos. The averaged arm is the
one to serve. See `CLAUDE.md` for the goal those numbers are judged against, and
`EggIC/CLAUDE_LOG.md` item T for the full result.

⚠️ Label smoothing caps the model's confidence near 0.911, so the averaged arm reaches
its zero-error point at **0.71**, not 0.90. The confidence *scale* is the thing to fix
before the 0.90 threshold in `CONFIDENT_AT` means what it says.

### Swapping the model

```bash
modal volume put eggic-models final_ema.pt v2/final_ema.pt
modal volume put eggic-models calibration_T1.json v2/calibration_T1.json
MODEL_VERSIONS=v1,dinov3,v2 modal deploy modal/app.py
```

`final_ema.pt` is preferred over `best.pt` when a run wrote both. Until the DINOv3
work merges to `EggIC`'s `main`, deploying also needs a source checkout new enough
to build it — `EGGIC_SRC=../EggIC-modal-eval/eggic-fgvc/src modal deploy modal/app.py`.
The deploy fails loudly rather than serving a checkout that cannot build DINOv3.

Old versions stay on the volume, so rollback is one redeploy. Every stored
submission records the `model_version` that served it, which is what makes two
models comparable on the same real photographs.

## What gets stored

Everything is in one Supabase project; `supabase/migrations/` is the whole schema.

| | holds | readable by |
|---|---|---|
| bucket `submissions` | every photo the live site identified, `YYYY-MM/<id>.jpg` | secret key only |
| table `submissions` | one row per photo: top species, confidence, all nine probabilities, each arm's answer, the tester's verdict | secret key only |
| bucket `test-images` | the staff-test photos, under random names | logged-in testers |
| table `test_images` | the true species of each test photo, and what the model said | secret key only |
| table `test_answers` | one row per tester per photo: the species they chose and an optional note | each tester, own rows only |

The stored photograph is what the browser sent: downscaled to 1024px on the long
edge, not the phone original. In `submissions`, `correct` is null until a tester
answers; `user_species` is the class they chose; `threshold` is the accept line in
force when the photo was served. Submissions from before Supabase were copied in
from the old Modal volume with `scripts/import_volume.py`.

### Pulling the data back

In the Supabase SQL editor:

```sql
-- the retraining queue
select * from submissions where correct = false order by created_at desc;

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

Single screen, no page scroll: photograph on the left, verdict and top three species
on the right, the remaining six behind a disclosure. Light by default, with a theme
toggle persisted in localStorage.

**Batches.** Up to 20 photographs at a time, identified one after another with
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
