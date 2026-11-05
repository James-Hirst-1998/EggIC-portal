"""Make the staff-test set match a CSV: upload the photos and write the answer key.
Re-runnable. The CSV is the whole set: photos already loaded (same source path) are
skipped, and photos no longer in it are removed — refused if anyone has answered one.

    python scripts/load_test_set.py test_set.csv

The CSV has path (relative to EggIC_data), species, specimen. Each photo is turned
upright, shrunk to 1024px on the long side and saved as JPEG q92 — exactly what the
public page sends the model — which also drops EXIF, GPS included. It is stored
under a random name, because some source names give the species away.
"""

from __future__ import annotations

import csv
import io
import sys
import uuid

from PIL import Image, ImageOps

from _supabase import DATA, SPECIES, call, upload, upsert

LONG_SIDE = 1024
BATCH = 50   # names per request, to keep URLs short


def prepare(path) -> bytes:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def remove(names: list[str]) -> None:
    """Drop photos from the set: answer-key rows first, then the stored files."""
    for i in range(0, len(names), BATCH):
        batch = names[i:i + BATCH]
        names_in = "in.(" + ",".join(f'"{n}"' for n in batch) + ")"
        answered = call("GET", f"/rest/v1/test_answers?select=image_path&image_path={names_in}").json()
        if answered:
            raise SystemExit(f"{len(answered)} answers are on photos this CSV drops; "
                             "keep those photos in the CSV")
        call("DELETE", f"/rest/v1/test_images?image_path={names_in}")
        call("DELETE", "/storage/v1/object/test-images", json={"prefixes": batch})


def main() -> None:
    rows = list(csv.DictReader(open(sys.argv[1])))
    bad = sorted({r["species"] for r in rows} - set(SPECIES))
    if bad:
        raise SystemExit(f"unknown species {bad}; expected one of {SPECIES}")
    missing = [r["path"] for r in rows if not (DATA / r["path"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} photos not found under {DATA}, e.g. {missing[:3]}")

    current = call("GET", "/rest/v1/test_images?select=image_path,source").json()
    loaded = {r["source"] for r in current}
    todo = [r for r in rows if r["path"] not in loaded]
    keep = {r["path"] for r in rows}
    gone = [r["image_path"] for r in current if r["source"] not in keep]
    print(f"{len(rows)} in the CSV, {len(rows) - len(todo)} already loaded, "
          f"{len(todo)} to add, {len(gone)} to remove")
    remove(gone)

    for i, r in enumerate(todo, 1):
        name = f"{uuid.uuid4().hex[:12]}.jpg"
        upload("test-images", name, prepare(DATA / r["path"]))
        upsert("test_images", [{"image_path": name, "true_species": r["species"],
                                "specimen": r["specimen"] or None, "source": r["path"]}])
        print(f"\r{i}/{len(todo)}", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
