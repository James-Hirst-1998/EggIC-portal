"""Copy submissions from the old eggic-submissions volume into Supabase. Re-runnable.

    modal volume get eggic-submissions 2026-08 ./inbox
    python scripts/import_volume.py ./inbox
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _supabase import upload, upsert


def row(s: dict) -> dict:
    p, fb = s["prediction"], s["feedback"]
    return {
        "id": s["id"],
        "created_at": s["ts"],
        "image_path": s["image_file"],
        "filename": s.get("original_filename"),
        "model_version": s["model_version"],
        "top": p["top"],
        "confidence": p["confidence"],
        "confident": p["confident"],
        "threshold": p["threshold"],
        "ranked": p["ranked"],
        "by_model": s.get("by_model"),   # only stored from September on
        "latency_ms": s.get("latency_ms"),
        "correct": fb["correct"],
        "user_species": fb["user_species"],
        "feedback_at": fb["ts"],
    }


def main() -> None:
    folder = Path(sys.argv[1])
    sidecars = sorted(folder.rglob("*.json"))
    if not sidecars:
        raise SystemExit(f"no sidecars under {folder}")
    for i, f in enumerate(sidecars, 1):
        s = json.loads(f.read_text())
        image = f.with_suffix(".jpg")
        if not image.is_file():
            raise SystemExit(f"{f} has no photo beside it")
        upload("submissions", s["image_file"], image.read_bytes())
        upsert("submissions", [row(s)])
        print(f"\r{i}/{len(sidecars)}", end="", flush=True)
    print(f"\nimported {len(sidecars)} submissions")


if __name__ == "__main__":
    main()
