"""Modal deployment of the egg-case classifier.

    modal deploy modal/app.py

Weights live on the ``eggic-models`` Volume, not in the image, so swapping the
served model is a volume put plus an env change — no rebuild:

    modal volume put eggic-models best.pt v2/best.pt
    MODEL_VERSION=v2 modal deploy modal/app.py

The model *definition* is imported from the sibling EggIC checkout, so the served
network cannot drift from the weights that produced it.

Every photo is segmented first (EggIC ``multispecies/segment.py``, SAM + keep head),
and the reply is always the same shape: a list of egg cases, each with its outline,
its answer and every class's score (README, "What the API returns"). Two or more found
cleanly → each is cropped and named on its own; more than ``MAX_CASES`` → none is
named; otherwise the whole photo is named.

Submissions are written to Supabase (photo and masks to the ``submissions`` bucket,
a row to ``submissions`` and one per egg case to ``cases``) only when the caller asks
for it. The schema is ``supabase/migrations/``.
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
                  os.environ.get("MODEL_VERSIONS", "s2_dinov2,s2_dinov3").split(",") if v.strip()]
CONFIDENT_AT = 0.95  # clear of the worst mistake seen on the held-out photos (93.7%)
MAX_CASES = 8  # more than this in one photo: "too_many", and nothing goes to the classifier
SAM_CKPT = "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything"
KEEP_BACKBONE = "vit_large_patch14_dinov2.lvd142m"   # multispecies.keep.Backbone.NAME

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
        "segment-anything==1.0",
    )
    .apt_install("wget")
    .run_commands(f"mkdir -p /root/sam && wget -q -O /root/sam/{SAM_CKPT} {SAM_URL}/{SAM_CKPT}")
    # The segmenter's frozen backbone, baked in so a cold start does not fetch 1.2 GB.
    .run_commands(f"python -c \"import timm; timm.create_model('{KEEP_BACKBONE}', "
                  "pretrained=True, num_classes=0, img_size=224)\"")
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
    multispecies = fgvc_src.parents[1] / "multispecies"
    if not (multispecies / "segment.py").is_file():
        raise RuntimeError(f"Cannot find the segmenter at {multispecies}: pull EggIC "
                           "(merged 2026-11-06) or point EGGIC_SRC at a checkout that has it.")
    image = (
        image
        .add_local_dir(fgvc_src, remote_path="/root/fgvc_src")
        .add_local_dir(multispecies, remote_path="/root/multispecies",
                       ignore=["tests", "__pycache__"])
        .add_local_file(HERE / "inference.py", remote_path="/root/inference.py")
    )

models = modal.Volume.from_name("eggic-models", create_if_missing=True)
api_key = modal.Secret.from_name("eggic-api-key")
# SUPABASE_URL and SUPABASE_SECRET_KEY. The secret key bypasses row-level security,
# so it lives here and never in Vercel or a page.
supabase = modal.Secret.from_name("eggic-supabase")

SUBMISSION_ID = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")
# What each egg case in the reply carries, in this order.
CASE_FIELDS = ("n", "box", "outline", "mask_path", "top", "top_name", "confidence",
               "confident", "ranked", "by_model")


def _supabase(method: str, path: str, **kwargs):
    """One call to the Supabase REST or Storage API as the service role."""
    import httpx

    key = os.environ["SUPABASE_SECRET_KEY"]
    headers = {"apikey": key, **kwargs.pop("headers", {})}
    resp = httpx.request(method, os.environ["SUPABASE_URL"].rstrip("/") + path,
                         headers=headers, timeout=20, **kwargs)
    resp.raise_for_status()
    return resp


def _outline(mask) -> list:
    """A mask's edge as polygons, each ``[[x, y], ...]`` in fractions of the photo."""
    import cv2
    import numpy as np

    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    biggest = max((cv2.contourArea(c) for c in contours), default=0)
    return [[[round(x / w, 4), round(y / h, 4)] for x, y in
             cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()]
            for c in contours if cv2.contourArea(c) >= 0.05 * biggest]


def _png(mask) -> bytes:
    """A mask as a black-and-white PNG the size of the photo."""
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("could not encode a mask")
    return buf.tobytes()


@app.cls(
    image=image,
    # fp32 ViT-L x2; the T4 was only viable at fp16, which DINOv3 cannot use. A10G is the
    # fallback: on 2026-11-17 an L4 took 27 minutes to schedule.
    gpu=["L4", "A10G"],
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
        self._load_segmenter()
        # The average is built from raw probabilities, so its display scale is its
        # own — it lives on the volume beside the arms.
        ens = Path("/models/ensemble/calibration_T1.json")
        self.ensemble_display_t = (json.loads(ens.read_text())["display_temperature"]
                                   if ens.is_file() else 1.0)

    def _load_segmenter(self):
        """SAM proposes masks, the keep head keeps whole egg cases (EggIC log AE/AF).

        The head and the crop fill are copied from the eggic-multispecies Volume to
        ``/models/segment`` so a retrain there cannot change what is served.
        """
        from multispecies import keep, segment
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        t0 = time.time()
        sam = sam_model_registry["vit_h"](checkpoint=f"/root/sam/{SAM_CKPT}").to("cuda")
        self.sam = SamAutomaticMaskGenerator(sam, **segment.PARAMS.amg_kwargs())
        self.keep_backbone = keep.Backbone(device="cuda")
        self.keep_head = keep.load_head("/models/segment/keep_head.npz")
        self.fill = json.loads(Path("/models/segment/meta.json").read_text())["fill_target"]
        print(f"loaded segmenter in {time.time() - t0:.1f}s (fill {self.fill})")

    def _segment(self, raw: bytes) -> dict:
        """The segmenter's verdict and, when it is clean, each egg case's box and mask.

        Only "ok" is trusted. More than ``MAX_CASES`` is "too_many" here even when the
        segmenter (which stops at 10) kept them all. Cases come top to bottom.
        """
        from multispecies import crops, segment

        t0 = time.time()
        img = segment.prepare(raw)
        seg = segment.segment(img, self.sam, self.keep_backbone, self.keep_head, self.fill)
        status = "too_many" if seg["status"] == "ok" and seg["count"] > MAX_CASES else seg["status"]
        out = {"status": status, "found": []}
        if status == "ok":
            shape = tuple(seg["size"])
            h, w = shape
            masks = [segment.unrle(c["mask"], shape) for c in seg["cases"]]
            for i, c in enumerate(seg["cases"]):
                x0, y0, x1, y1 = c["box"]
                out["found"].append({
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                    "mask": masks[i],
                    # A lone case is named on the whole photo, so it needs no crop.
                    "crop": crops.crop_instance(img, masks[i], tuple(c["box"]), fill=self.fill,
                                                others=[m for j, m in enumerate(masks) if j != i])
                            if len(masks) > 1 else None})
        out["seconds"] = round(time.time() - t0, 1)
        return out

    @staticmethod
    def _check_key(req: dict) -> None:
        from fastapi import HTTPException

        if req.get("key") != os.environ["EGGIC_API_KEY"]:
            raise HTTPException(status_code=401, detail="bad key")

    def _store(self, raw: bytes, filename: str | None, result: dict, masks: list) -> str:
        """The photo and each case's mask to the bucket, one row per photo and per case."""
        now = datetime.now(timezone.utc)
        sid = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        folder = now.strftime("%Y-%m")

        _supabase("POST", f"/storage/v1/object/submissions/{folder}/{sid}.jpg",
                  content=raw, headers={"Content-Type": "image/jpeg"})
        for case, mask in zip(result["cases"], masks):
            if mask is not None:
                case["mask_path"] = f"{folder}/{sid}/{case['n']}.png"
                _supabase("POST", f"/storage/v1/object/submissions/{case['mask_path']}",
                          content=_png(mask), headers={"Content-Type": "image/png"})
        _supabase("POST", "/rest/v1/submissions", headers={"Prefer": "return=minimal"}, json={
            "id": sid,
            "created_at": now.isoformat(),
            "image_path": f"{folder}/{sid}.jpg",
            "filename": filename,
            "model_version": result["model_version"],
            "threshold": CONFIDENT_AT,
            "latency_ms": result["latency_ms"],
            "segment_status": result["segmenter"]["status"],
            "case_count": result["count"],
        })
        if result["cases"]:
            _supabase("POST", "/rest/v1/cases", headers={"Prefer": "return=minimal"}, json=[{
                "submission_id": sid,
                "n": c["n"],
                "box": c["box"],
                "outline": c["outline"],
                "mask_path": c["mask_path"],
                "top": c["top"],
                "confidence": c["confidence"],
                "confident": c["confident"],
                "ranked": {r["key"]: round(r["prob"], 6) for r in c["ranked"]},
                "by_model": {m: {"top": r["top"], "confidence": round(r["confidence"], 6)}
                             for m, r in c["by_model"].items()},
            } for c in result["cases"]])
        return sid

    @modal.fastapi_endpoint(method="POST", docs=False)
    def predict(self, req: dict):
        """One photo in, its egg cases out: README, "What the API returns"."""
        from fastapi import HTTPException
        from inference import load_rgb

        self._check_key(req)

        try:
            raw = base64.b64decode(str(req["image"]).split(",")[-1])
            image = load_rgb(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400,
                                detail=f"could not decode image: {type(exc).__name__}: {exc}")

        # Default to every loaded arm, not the first: a caller that names no models
        # must still get the average, which is the only configuration measured as
        # error-free at the accept line.
        wanted = req.get("models") or list(self.versions)
        unknown = [m for m in wanted if m not in self.clfs]
        if unknown:
            raise HTTPException(status_code=400,
                                detail=f"unknown model(s) {unknown}; loaded: {self.versions}")
        primary = req.get("primary") or ("ensemble" if len(wanted) > 1 else wanted[0])
        if primary not in (["ensemble"] if len(wanted) > 1 else []) + wanted:
            raise HTTPException(status_code=400, detail=f"primary {primary!r} was not run")

        t0 = time.time()
        seg = (self._segment(raw) if req.get("segment", True)
               else {"status": "skipped", "seconds": 0.0, "found": []})
        found = seg["found"]
        if seg["status"] == "too_many":
            found = []  # too many to name one by one: the reply says so and names none
        elif len(found) < 2:
            # One case or none found cleanly: name the whole photo, as the model was trained.
            found = [found[0] if found else {"box": None, "mask": None, "crop": None}]
            found[0]["crop"] = image
        cases = [{"box": f["box"], "mask": f["mask"],
                  **self._answer(f["crop"], wanted, primary)} for f in found]

        # A case the model is sure is not an egg case is not reported: an empty list
        # means none was found, and no_egg_case says how sure that call was.
        none = [c for c in cases if c["confident"] and c["top"] == "NoEggCase"]
        cases = [c for c in cases if not (c["confident"] and c["top"] == "NoEggCase")]
        masks = [c.pop("mask") for c in cases]
        for n, (c, m) in enumerate(zip(cases, masks), 1):
            c.update(n=n, outline=_outline(m) if m is not None else None, mask_path=None)

        result = {
            "id": None,
            "model_version": primary,
            "threshold": CONFIDENT_AT,
            "latency_ms": round((time.time() - t0) * 1000),
            "segmenter": {"status": seg["status"], "seconds": seg["seconds"], "max_cases": MAX_CASES},
            "count": len(cases),
            "no_egg_case": none[0]["confidence"] if none and not cases else None,
            "cases": [{k: c[k] for k in CASE_FIELDS} for c in cases],
        }
        # Off for local runs; the deployed site turns it on.
        if req.get("store"):
            result["id"] = self._store(raw, req.get("filename"), result, masks)
        return result

    def _answer(self, image, wanted: list, primary: str) -> dict:
        """One egg case's answer, with every class's score and each arm's call."""
        r = self._classify(image, wanted, primary)
        return {"top": r["top"], "top_name": r["top_name"], "confidence": r["confidence"],
                "confident": r["confident"], "ranked": r["ranked"],
                "by_model": {m: {"top": a["top"], "top_name": a["top_name"],
                                 "confidence": a["confidence"]}
                             for m, a in r["by_model"].items()}}

    def _classify(self, image, wanted: list, primary: str) -> dict:
        """Every wanted arm, their average, the shown scale and the accept line."""
        from fastapi import HTTPException
        from inference import apply_display_temperature

        by_model = {m: self.clfs[m].predict(image) for m in wanted}
        if len(by_model) > 1:
            by_model["ensemble"] = self._average(by_model, wanted)

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
        result["by_model"] = by_model
        result["confident"] = result["confidence"] >= CONFIDENT_AT
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
    """Record whether one egg case's answer was right, and what it actually was."""
    from fastapi import HTTPException

    if req.get("key") != os.environ["EGGIC_API_KEY"]:
        raise HTTPException(status_code=401, detail="bad key")

    sid, n = str(req.get("id") or ""), req.get("n")
    if not SUBMISSION_ID.fullmatch(sid):
        raise HTTPException(status_code=400, detail=f"bad submission id {sid!r}")
    if type(n) is not int or n < 1:
        raise HTTPException(status_code=400, detail=f"bad case number {n!r}")

    rows = _supabase(
        "PATCH", f"/rest/v1/cases?submission_id=eq.{sid}&n=eq.{n}",
        headers={"Prefer": "return=representation"},
        json={"correct": req.get("correct"), "user_species": req.get("user_species"),
              "feedback_at": datetime.now(timezone.utc).isoformat()},
    ).json()
    if not rows:
        raise HTTPException(status_code=404, detail=f"no case {n} in submission {sid}")
    return {"stored": True, "id": sid, "n": n}
