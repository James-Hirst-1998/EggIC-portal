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
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
CONFIDENT_AT = 0.85  # above this we name a species; below, the UI shows the field

app = modal.App("eggic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pinned to the env that produced the checkpoint. timm especially: 0.9.16 is
    # not interchangeable with 1.0.x here — build.py imports param_groups_layer_decay
    # from timm.optim.optim_factory, which moved in 1.0.
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "timm==0.9.16",
        "albumentations==2.0.8",
        "opencv-python-headless==4.11.0.86",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "fastapi[standard]==0.115.12",
    )
    .env({"PYTHONPATH": "/root:/root/fgvc_src", "MODEL_VERSION": MODEL_VERSION})
)

# This module is imported inside the container as well as locally, where the
# sibling checkout does not exist and the local files are already in place.
if modal.is_local():
    fgvc_src = HERE.parents[1] / "EggIC" / "eggic-fgvc" / "src"
    if not (fgvc_src / "eggic_fgvc").is_dir():
        raise RuntimeError(
            f"Cannot find eggic_fgvc at {fgvc_src}. Deploying expects EggIC/ and "
            "EggIC-portal/ as siblings under code/personal."
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
    gpu="T4",
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

        t0 = time.time()
        self.version = os.environ["MODEL_VERSION"]
        self.clf = Classifier(Path("/models") / self.version, device="cuda")
        print(f"loaded {self.version} in {time.time() - t0:.1f}s "
              f"({self.clf.image_size}px, T={self.clf.temperature}, "
              f"{len(self.clf.views)} TTA views)")

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
            "model_version": self.version,
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

        t0 = time.time()
        result = self.clf.predict(image)
        result["latency_ms"] = round((time.time() - t0) * 1000)
        result["model_version"] = self.version
        result["confident"] = result["confidence"] >= CONFIDENT_AT
        result["threshold"] = CONFIDENT_AT

        # Off for local runs; the deployed site turns it on.
        result["id"] = (self._store(raw, req.get("filename"), result)
                        if req.get("store") else None)
        return result


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
