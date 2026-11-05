"""Modal deployment of the egg-case classifier.

    modal deploy modal/app.py

Weights live on the ``eggic-models`` Volume, not in the image, so swapping the
served model is a volume put plus an env change — no rebuild:

    modal volume put eggic-models best.pt v2/best.pt
    MODEL_VERSION=v2 modal deploy modal/app.py

The model *definition* is imported from the sibling EggIC checkout, so the served
network cannot drift from the weights that produced it.

Submissions are written to Supabase (photo to the ``submissions`` bucket, a row
to the ``submissions`` table) only when the caller asks for it. The schema is
``supabase/migrations/``.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Comma-separated: every one is loaded into the same container, and a request
# picks which to run. Two ViT-L at fp16 fit a T4 with room to spare.
MODEL_VERSIONS = [v.strip() for v in
                  os.environ.get("MODEL_VERSIONS", "v1,dinov3").split(",") if v.strip()]
CONFIDENT_AT = 0.875  # 4 points clear of the worst mistake seen on the held-out photos

app = modal.App("eggic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # timm 1.0.28 is required by the DINOv3 backbone, and it moved
    # param_groups_layer_decay out of timm.optim.optim_factory — so the pin and
    # the mounted eggic_fgvc source have to travel together (checked below).
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "timm==1.0.28",
        "albumentations==2.0.8",
        "opencv-python-headless==4.11.0.86",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "fastapi[standard]==0.115.12",
    )
    .env({"PYTHONPATH": "/root:/root/fgvc_src",
          "MODEL_VERSIONS": ",".join(MODEL_VERSIONS)})
)

# This module is imported inside the container as well as locally, where the
# sibling checkout does not exist and the local files are already in place.
if modal.is_local():
    # EGGIC_SRC points at a checkout newer than main while the DINOv3 work is on
    # a branch; once it merges, the default is right again.
    fgvc_src = Path(os.environ.get(
        "EGGIC_SRC", HERE.parents[1] / "EggIC" / "eggic-fgvc" / "src")).expanduser()
    if not (fgvc_src / "eggic_fgvc").is_dir():
        raise RuntimeError(
            f"Cannot find eggic_fgvc at {fgvc_src}. Deploying expects EggIC/ and "
            "EggIC-portal/ as siblings under code/personal, or EGGIC_SRC set."
        )
    build_py = (fgvc_src / "eggic_fgvc" / "models" / "build.py").read_text()
    if "from timm.optim.optim_factory import" in build_py:
        raise RuntimeError(
            f"{fgvc_src} is a pre-timm-1.0 checkout: build.py imports "
            "param_groups_layer_decay from timm.optim.optim_factory, which the "
            "pinned timm 1.0.28 no longer exposes, and it cannot build DINOv3. "
            "Set EGGIC_SRC to a checkout with the DINOv3 backbone."
        )
    image = (
        image
        .add_local_dir(fgvc_src, remote_path="/root/fgvc_src")
        .add_local_file(HERE / "inference.py", remote_path="/root/inference.py")
    )

models = modal.Volume.from_name("eggic-models", create_if_missing=True)
api_key = modal.Secret.from_name("eggic-api-key")
# SUPABASE_URL and SUPABASE_SECRET_KEY. The secret key bypasses row-level security,
# so it lives here and never in Vercel or a page.
supabase = modal.Secret.from_name("eggic-supabase")

SUBMISSION_ID = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")


def _supabase(method: str, path: str, **kwargs):
    """One call to the Supabase REST or Storage API as the service role."""
    import httpx

    key = os.environ["SUPABASE_SECRET_KEY"]
    headers = {"apikey": key, **kwargs.pop("headers", {})}
    resp = httpx.request(method, os.environ["SUPABASE_URL"].rstrip("/") + path,
                         headers=headers, timeout=20, **kwargs)
    resp.raise_for_status()
    return resp


@app.cls(
    image=image,
    gpu="L4",   # fp32 ViT-L x2; the T4 was only viable at fp16, which DINOv3 cannot use
    volumes={"/models": models},
    secrets=[api_key, supabase],
    # Idle GPU time is the main cost here, not the inference. 60s covers the gaps
    # inside a review session; anything longer just burns credits after you stop.
    scaledown_window=60,
    timeout=120,
)
class Model:
    @modal.enter()
    def load(self):
        from inference import Classifier

        self.versions = [v for v in os.environ["MODEL_VERSIONS"].split(",") if v]
        self.clfs = {}
        for version in self.versions:
            t0 = time.time()
            self.clfs[version] = Classifier(Path("/models") / version, device="cuda")
            clf = self.clfs[version]
            print(f"loaded {version} in {time.time() - t0:.1f}s "
                  f"({clf.image_size}px, T={clf.temperature}, {len(clf.views)} TTA views)")
        self.version = self.versions[0]
        # The average is built from raw probabilities, so its display scale is its
        # own — it lives on the volume beside the arms.
        ens = Path("/models/ensemble/calibration_T1.json")
        self.ensemble_display_t = (json.loads(ens.read_text())["display_temperature"]
                                   if ens.is_file() else 1.0)

    @staticmethod
    def _check_key(req: dict) -> None:
        from fastapi import HTTPException

        if req.get("key") != os.environ["EGGIC_API_KEY"]:
            raise HTTPException(status_code=401, detail="bad key")

    def _store(self, raw: bytes, filename: str | None, result: dict) -> str:
        """The photo to the submissions bucket, everything else to one row."""
        now = datetime.now(timezone.utc)
        sid = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        image_path = f"{now.strftime('%Y-%m')}/{sid}.jpg"

        _supabase("POST", f"/storage/v1/object/submissions/{image_path}",
                  content=raw, headers={"Content-Type": "image/jpeg"})
        _supabase("POST", "/rest/v1/submissions", headers={"Prefer": "return=minimal"}, json={
            "id": sid,
            "created_at": now.isoformat(),
            "image_path": image_path,
            "filename": filename,
            "model_version": result.get("model_version", self.version),
            "top": result["top"],
            "confidence": result["confidence"],
            "confident": result["confident"],
            "threshold": CONFIDENT_AT,
            "ranked": {r["key"]: round(r["prob"], 6) for r in result["ranked"]},
            "by_model": {m: {"top": r["top"], "confidence": round(r["confidence"], 6)}
                         for m, r in result.get("by_model", {}).items()},
            "latency_ms": result["latency_ms"],
        })
        return sid

    @modal.fastapi_endpoint(method="POST", docs=False)
    def predict(self, req: dict):
        from fastapi import HTTPException
        from inference import load_rgb

        self._check_key(req)

        try:
            raw = base64.b64decode(str(req["image"]).split(",")[-1])
            image = load_rgb(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400,
                                detail=f"could not decode image: {type(exc).__name__}: {exc}")

        from inference import apply_display_temperature

        # Default to every loaded arm, not the first: a caller that names no models
        # (the old front-end does not) must still get the average, which is the only
        # configuration measured as error-free at the accept line.
        wanted = req.get("models") or list(self.versions)
        unknown = [m for m in wanted if m not in self.clfs]
        if unknown:
            raise HTTPException(status_code=400,
                                detail=f"unknown model(s) {unknown}; loaded: {self.versions}")

        t0 = time.time()
        by_model = {m: self.clfs[m].predict(image) for m in wanted}
        if len(by_model) > 1:
            by_model["ensemble"] = self._average(by_model, wanted)

        # The ensemble is the arm the held-out evaluation actually favours, so it
        # leads whenever more than one model ran.
        primary = req.get("primary") or ("ensemble" if len(by_model) > 1 else wanted[0])
        if primary not in by_model:
            raise HTTPException(status_code=400, detail=f"primary {primary!r} was not run")

        for name, r in by_model.items():
            if not all(math.isfinite(x["prob"]) for x in r["ranked"]):
                raise HTTPException(status_code=500,
                                    detail=f"{name} returned non-finite probabilities")

        # Display scale last: averaging must happen on the raw probabilities.
        for name, r in by_model.items():
            t = (self.ensemble_display_t if name == "ensemble"
                 else self.clfs[name].display_temperature)
            r["ranked"] = apply_display_temperature(r["ranked"], t)
            r["confidence"] = r["ranked"][0]["prob"]
            r["display_temperature"] = t

        result = dict(by_model[primary])
        result["latency_ms"] = round((time.time() - t0) * 1000)
        result["model_version"] = primary
        result["models_run"] = list(by_model)
        result["by_model"] = by_model
        result["confident"] = result["confidence"] >= CONFIDENT_AT
        result["threshold"] = CONFIDENT_AT

        # Off for local runs; the deployed site turns it on.
        result["id"] = (self._store(raw, req.get("filename"), result)
                        if req.get("store") else None)
        return result

    @staticmethod
    def _average(by_model: dict, names: list) -> dict:
        """Softmax average across arms — +7 of 224 held-out photos over either alone."""
        totals = {}
        for m in names:
            for r in by_model[m]["ranked"]:
                totals[r["key"]] = totals.get(r["key"], 0.0) + r["prob"] / len(names)
        ranked = sorted(
            ({**next(r for r in by_model[names[0]]["ranked"] if r["key"] == k), "prob": v}
             for k, v in totals.items()), key=lambda r: -r["prob"])
        return {"top": ranked[0]["key"], "top_name": ranked[0]["name"],
                "confidence": ranked[0]["prob"],
                "reliability": "softmax average of " + " + ".join(names),
                "n_views": sum(by_model[m]["n_views"] for m in names),
                "ranked": ranked}


# Feedback updates one row and needs nothing else — no model, no GPU.
# Kept off the Model class deliberately: as a method it inherited @modal.enter(),
# so saving a one-line answer paid a GPU cold start and 1.2 GB of weight loading.
# It stays on Modal rather than moving to Vercel so the Supabase secret key lives
# in one place only.
feedback_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.12")
)


@app.function(
    image=feedback_image,
    secrets=[api_key, supabase],
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def feedback(req: dict):
    """Record whether the call was right, and what the species actually was."""
    from fastapi import HTTPException

    if req.get("key") != os.environ["EGGIC_API_KEY"]:
        raise HTTPException(status_code=401, detail="bad key")

    sid = str(req.get("id") or "")
    if not SUBMISSION_ID.fullmatch(sid):
        raise HTTPException(status_code=400, detail=f"bad submission id {sid!r}")

    rows = _supabase(
        "PATCH", f"/rest/v1/submissions?id=eq.{sid}",
        headers={"Prefer": "return=representation"},
        json={"correct": req.get("correct"), "user_species": req.get("user_species"),
              "feedback_at": datetime.now(timezone.utc).isoformat()},
    ).json()
    if not rows:
        raise HTTPException(status_code=404, detail=f"no submission {sid}")
    return {"stored": True, "id": sid}
