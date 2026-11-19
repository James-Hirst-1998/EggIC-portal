"""Ask the live model about every test photo it hasn't answered yet, and save its
answer into the key. Each photo is the same 1024px JPEG the staff see.

    python scripts/score_test_set.py

Runs the deployed GPU endpoint: about 2 s a photo once warm, plus one cold start.
Nothing is stored as a submission. Each test photo holds one egg case, so it is
named whole (no segmenting), as the staff saw it.
"""

from __future__ import annotations

import base64

import requests

from _supabase import call, env


def main() -> None:
    try:
        endpoint = env("EGGIC_PREDICT_URL")
    except SystemExit:
        endpoint = env("EGGIC_ENDPOINT")   # the older name local/server.py also accepts
    key = env("EGGIC_API_KEY")
    todo = call("GET", "/rest/v1/test_images?select=image_path&model_top=is.null").json()
    print(f"{len(todo)} photos to score")
    for i, r in enumerate(todo, 1):
        jpeg = call("GET", f"/storage/v1/object/test-images/{r['image_path']}").content
        resp = requests.post(endpoint, timeout=300, json={
            "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            "key": key, "store": False, "segment": False,
        })
        resp.raise_for_status()
        result = resp.json()
        # An empty list is the model sure the photo holds no egg case: a confident answer.
        top, conf = ((result["cases"][0]["top"], result["cases"][0]["confidence"])
                     if result["cases"] else ("NoEggCase", result["no_egg_case"]))
        call("PATCH", f"/rest/v1/test_images?image_path=eq.{r['image_path']}",
             json={"model_top": top, "model_confidence": conf})
        print(f"\r{i}/{len(todo)}", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
