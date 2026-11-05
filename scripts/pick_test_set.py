"""Pick the staff-test photos and write them as a CSV for load_test_set.py.

    python scripts/pick_test_set.py > test_set.csv

About 150 photos, weighted to the species the model finds hardest (QUOTA: Thornback
and Blonde most, Cuckoo and Flapper least — from the 224 held-out photos and the
real submissions with feedback). Only photos no model has trained on: the
benchmark, the Shark Trust export's test tiers, and the photos testers submitted
through the portal.

Within a species, hard photos go in first, up to HARD_SHARE of its quota: ones the
live model got wrong or answered below the line, and damaged cases. The rest is a
random draw across the three sources, for range. Near-identical photos are dropped,
and no more than two come from one sighting.
"""

from __future__ import annotations

import csv
import random
import re
import sys
from collections import defaultdict

from PIL import Image, ImageOps

from _supabase import DATA, SPECIES, call

QUOTA = {"ThornbackRay": 26, "BlondeRay": 26, "SpottedRay": 22, "UndulateRay": 18,
         "SmallEyedRay": 16, "SmallSpottedCatshark": 12, "Nursehound": 12,
         "FlapperSkate": 9, "CuckooRay": 9}
HARD_SHARE = 0.6
PER_SPECIMEN = 2
CONFIDENT_AT = 0.875   # the served line, modal/app.py

BENCHMARK = {"Blonde": "BlondeRay", "Cuckoo": "CuckooRay", "Flapper": "FlapperSkate",
             "Nursehound": "Nursehound", "Small-eyed": "SmallEyedRay",
             "SSC": "SmallSpottedCatshark", "Spotted": "SpottedRay",
             "Thornback": "ThornbackRay", "Undulate": "UndulateRay"}
EXPORT = DATA / "ST_export_2026-11"
FIELD = ["2026-08-06 tested images", "2026-09-03 tested images"]
# Reasons a reviewer gave that mean "damaged but still an egg case to name".
DAMAGE = re.compile(r"crumpled|torn|folded|hole|missing|frayed|split|shredded|hidden by weed")
# Photos ruled out by eye, path relative to EggIC_data: they test something other
# than naming one egg case.
EXCLUDE = {
    "2026-08-06 tested images/ThornbackRay_20260806T151032-b10f859b.jpg": "species written in the photo",
    "2026-08-06 tested images/ThornbackRay_20260806T143733-5e9c5f9c.jpg": "two egg cases",
    "2026-08-06 tested images/Nursehound_20260806T144939-72b750c0.jpg": "two egg cases",
    "2026-08-06 tested images/ThornbackRay_20260806T135034-dc7dc26d.jpg": "two egg cases",
}


def benchmark() -> list[dict]:
    out = []
    for p in sorted((DATA / "benchmark_images").glob("*.jpg")):
        prefix = p.stem.rsplit("_", 1)[0]
        if prefix in BENCHMARK:   # rare species are not in the nine
            out.append({"path": p, "species": BENCHMARK[prefix], "specimen": "",
                        "source": "benchmark", "why": ""})
    return out


def export() -> list[dict]:
    flags = {r["key"]: r for r in csv.DictReader(open(EXPORT / "review/ai_flags.csv"))}
    rejudged = {r["key"]: r["reason"]
                for r in csv.DictReader(open(EXPORT / "review/rejudge_2026-11-05_keys.csv"))}
    out = []
    for r in csv.DictReader(open(EXPORT / "manifest.csv")):
        if r["tier"] not in ("test", "test_dup") or r["class9"] not in SPECIES:
            continue
        flag = flags.get(r["id"], {})
        reason = rejudged.get(r["id"], "") if flag.get("flag") == "ok" else flag.get("reason", "")
        if flag.get("flag") != "ok" and not DAMAGE.search(reason):
            continue   # unreviewed, or bad for a reason that leaves nothing to name
        out.append({"path": EXPORT / r["small"], "species": r["class9"],
                    "specimen": r["sightingId"], "source": "shark trust",
                    "why": reason if DAMAGE.search(reason) else ""})
    return out


def field() -> list[dict]:
    """Portal submissions, with the true species from the tester's feedback.

    The files are named after the model's answer, not the truth, so the name is
    never used as the label. A photo nobody gave feedback on has no label and is
    left out, as is one whose species is outside the nine.
    """
    served = {r["id"]: r for r in call(
        "GET", "/rest/v1/submissions?select=id,top,confidence,correct,user_species").json()}
    out = []
    for folder in FIELD:
        for p in sorted((DATA / folder).glob("*.jpg")):
            s = served.get(p.stem.split("_", 1)[1])
            if s is None:
                raise SystemExit(f"{p.name} has no row in submissions; run import_volume.py")
            species = s["top"] if s["correct"] else s["user_species"]
            if species not in SPECIES:
                continue
            why = (f"model said {s['top']}" if not s["correct"]
                   else f"model unsure ({s['confidence']:.0%})" if s["confidence"] < CONFIDENT_AT
                   else "")
            out.append({"path": p, "species": species, "specimen": "",
                        "source": "portal", "why": why})
    return out


def dhash(path) -> int:
    im = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((9, 8))
    px = list(im.getdata())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] < px[(i // 8) * 9 + i % 8 + 1])


def main() -> None:
    pool, seen = [], []
    for c in benchmark() + export() + field():
        h = dhash(c["path"])
        if any(bin(h ^ s).count("1") <= 6 for s in seen):
            continue   # the same shot twice
        seen.append(h)
        pool.append(c)

    rng = random.Random(0)
    out = csv.writer(sys.stdout)
    out.writerow(["path", "species", "specimen", "source", "why"])
    for species, quota in QUOTA.items():
        mine = [c for c in pool if c["species"] == species]
        rng.shuffle(mine)
        # Dropped after the shuffle, so ruling a photo out leaves the rest of the draw as it was.
        mine = [c for c in mine if str(c["path"].relative_to(DATA)) not in EXCLUDE]
        hard = [c for c in mine if c["why"]]
        # Easy photos dealt round-robin by source, so each species gets a mix.
        by_source = defaultdict(list)
        for c in mine:
            if not c["why"]:
                by_source[c["source"]].append(c)
        easy = [c for group in zip(*by_source.values()) for c in group]
        easy += [c for c in mine if not c["why"] and c not in easy]

        picked, per_specimen = [], defaultdict(int)
        for c in hard[:round(quota * HARD_SHARE)] + easy + hard:
            if len(picked) == quota:
                break
            if c in picked or (c["specimen"] and per_specimen[c["specimen"]] >= PER_SPECIMEN):
                continue
            per_specimen[c["specimen"]] += 1
            picked.append(c)
        if len(picked) < quota:
            raise SystemExit(f"{species}: only {len(picked)} photos for a quota of {quota}")
        for c in picked:
            out.writerow([c["path"].relative_to(DATA), species, c["specimen"],
                          c["source"], c["why"]])


if __name__ == "__main__":
    main()
