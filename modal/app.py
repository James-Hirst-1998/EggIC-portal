"""Modal deployment of the egg-case classifier.

    modal deploy modal/app.py

Weights live on the ``eggic-models`` Volume, not in the image, so swapping the
served model is a volume put plus an env change — no rebuild:

    modal volume put eggic-models best.pt v2/best.pt
    MODEL_VERSION=v2 modal deploy modal/app.py

The model *definition* is imported from the sibling EggIC checkout, so the served
network cannot drift from the weights that produced it.

Submissions are written to the ``eggic-submissions`` Volume only when the caller
asks for it. The on-disk schema is documented in the README.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
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
CONFIDENT_AT = 0.85  # above this we name a species; below, the UI shows the field

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
submissions = modal.Volume.from_name("eggic-submissions", create_if_missing=True)
api_key = modal.Secret.from_name("eggic-api-key")


@app.cls(
    image=image,
    gpu="L4",   # fp32 ViT-L x2; the T4 was only viable at fp16, which DINOv3 cannot use
    volumes={"/models": models, "/submissions": submissions},
    secrets=[api_key],
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

    @staticmethod
    def _check_key(req: dict) -> None:
        from fastapi import HTTPException

        if req.get("key") != os.environ["EGGIC_API_KEY"]:
            raise HTTPException(status_code=401, detail="bad key")

    def _store(self, raw: bytes, filename: str | None, result: dict) -> str:
        """One image plus one JSON sidecar per submission.

        Deliberately no index file: two people submitting at the same moment would
        write over each other's copy of it. The directory listing is the index.
        """
        now = datetime.now(timezone.utc)
        sid = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        month = now.strftime("%Y-%m")
        folder = Path("/submissions") / month
        folder.mkdir(parents=True, exist_ok=True)

        (folder / f"{sid}.jpg").write_bytes(raw)
        (folder / f"{sid}.json").write_text(json.dumps({
            "id": sid,
            "ts": now.isoformat(),
            "image_file": f"{month}/{sid}.jpg",
            "original_filename": filename,
            "model_version": result.get("model_version", self.version),
            "models_run": result.get("models_run", [self.version]),
            "by_model": {m: {"top": r["top"], "confidence": round(r["confidence"], 6)}
                         for m, r in result.get("by_model", {}).items()},
            "prediction": {
                "top": result["top"],
                "top_name": result["top_name"],
                "confidence": result["confidence"],
                "confident": result["confident"],
                "threshold": CONFIDENT_AT,
                "ranked": {r["key"]: round(r["prob"], 6) for r in result["ranked"]},
            },
            "latency_ms": result["latency_ms"],
            # Filled in by /feedback; null means nobody has answered yet.
            "feedback": {"correct": None, "user_species": None, "ts": None},
        }, indent=2))
        submissions.commit()
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

        wanted = req.get("models") or [self.version]
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


# Feedback edits a small JSON file and needs nothing else — no model, no GPU.
# Kept off the Model class deliberately: as a method it inherited @modal.enter(),
# so saving a one-line answer paid a GPU cold start and 1.2 GB of weight loading.
feedback_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.12")
)


@app.function(
    image=feedback_image,
    volumes={"/submissions": submissions},
    secrets=[api_key],
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def feedback(req: dict):
    """Record whether the call was right, and what the species actually was."""
    from fastapi import HTTPException

    if req.get("key") != os.environ["EGGIC_API_KEY"]:
        raise HTTPException(status_code=401, detail="bad key")

    sid = str(req.get("id") or "")
    if not sid:
        raise HTTPException(status_code=400, detail="no submission id")

    # Another container wrote this file, so the local view must be refreshed.
    submissions.reload()
    matches = list(Path("/submissions").glob(f"*/{sid}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"no submission {sid}")

    sidecar = json.loads(matches[0].read_text())
    sidecar["feedback"] = {
        "correct": req.get("correct"),
        "user_species": req.get("user_species"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    matches[0].write_text(json.dumps(sidecar, indent=2))
    submissions.commit()
    return {"stored": True, "id": sid}
