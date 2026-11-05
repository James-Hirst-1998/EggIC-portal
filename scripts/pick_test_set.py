"""Pick the staff-test photos and write them as a CSV for load_test_set.py.

    python scripts/pick_test_set.py > test_set.csv

Every nine-species benchmark photo (held out from all training), plus up to 12 per
species from the Shark Trust export's moderated test tier. Only photos that passed
review (review/ai_flags.csv == ok) are used, so test_dup shots of the same egg come
in only where they were checked. Sightings with more than one photo are taken first,
so the set has some eggs seen from more than one angle.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

from _supabase import DATA, SPECIES

BENCHMARK = {"Blonde": "BlondeRay", "Cuckoo": "CuckooRay", "Flapper": "FlapperSkate",
             "Nursehound": "Nursehound", "Small-eyed": "SmallEyedRay",
             "SSC": "SmallSpottedCatshark", "Spotted": "SpottedRay",
             "Thornback": "ThornbackRay", "Undulate": "UndulateRay"}
PER_SPECIES = 12
EXPORT = DATA / "ST_export_2026-11"


def main() -> None:
    out = csv.writer(sys.stdout)
    out.writerow(["path", "species", "specimen"])

    bench = sorted((DATA / "benchmark_images").glob("*.jpg"))
    for p in bench:
        prefix = p.stem.rsplit("_", 1)[0]
        if prefix in BENCHMARK:   # rare species are not in the nine
            out.writerow([p.relative_to(DATA), BENCHMARK[prefix], ""])

    ok = {r["key"] for r in csv.DictReader(open(EXPORT / "review/ai_flags.csv"))
          if r["flag"] == "ok"}
    sightings = defaultdict(list)   # (species, sighting) -> photos
    for r in csv.DictReader(open(EXPORT / "manifest.csv")):
        if r["id"] in ok and r["tier"] in ("test", "test_dup") and r["class9"] in SPECIES:
            sightings[(r["class9"], r["sightingId"])].append(r["small"])

    rng = random.Random(0)
    for species in SPECIES:
        mine = [(k[1], v) for k, v in sightings.items() if k[0] == species]
        rng.shuffle(mine)
        mine.sort(key=lambda s: -len(s[1]))   # several angles first; stable, so still shuffled
        taken = 0
        for sighting, photos in mine:
            if taken + len(photos) > PER_SPECIES:
                continue
            for small in sorted(photos):
                out.writerow([(EXPORT / small).relative_to(DATA), species, sighting])
            taken += len(photos)


if __name__ == "__main__":
    main()
