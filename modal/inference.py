"""Eval-path inference for a trained eggic-fgvc checkpoint.

Mirrors the preprocessing in ``EggIC_models/viewer/infer.py`` — pad-to-square
(never crop, never squash: size and aspect ratio are diagnostic), ImageNet
normalisation, TTA over the config's scales x hflip. The model definition is
imported from ``eggic_fgvc`` rather than copied, so the served network cannot
drift from the weights.

Runs inside the Modal container, where the checkpoint lives on a Volume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # real phone photos are sometimes truncated

from eggic_fgvc.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from eggic_fgvc.models.build import build_model

SPECIES_INFO: Dict[str, Tuple[str, str]] = {
    "BlondeRay": ("Blonde Ray", "Raja brachyura"),
    "CuckooRay": ("Cuckoo Ray", "Leucoraja naevus"),
    "FlapperSkate": ("Flapper Skate", "Dipturus intermedius"),
    "Nursehound": ("Nursehound", "Scyliorhinus stellaris"),
    "SmallEyedRay": ("Small-eyed Ray", "Raja microocellata"),
    "SmallSpottedCatshark": ("Small-spotted Catshark", "Scyliorhinus canicula"),
    "SpottedRay": ("Spotted Ray", "Raja montagui"),
    "ThornbackRay": ("Thornback Ray", "Raja clavata"),
    "UndulateRay": ("Undulate Ray", "Raja undulata"),
}


def _tta_transform(image_size: int, scale: float, hflip: bool) -> A.Compose:
    steps = [
        A.LongestMaxSize(max_size=max(1, round(scale * image_size)), interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
    ]
    if hflip:
        steps.append(A.HorizontalFlip(p=1.0))
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


def load_rgb(path_or_bytes) -> np.ndarray:
    """HxWx3 uint8 RGB. EXIF is deliberately not applied — training and
    ``eggic_fgvc.evaluate`` both use a bare convert("RGB"), and honouring EXIF
    here costs 2 of 101 benchmark photos (84.2% -> 82.2%)."""
    return np.array(Image.open(path_or_bytes).convert("RGB"))


class Classifier:
    def __init__(self, run_dir: Path, device: str = "cuda", dtype: torch.dtype | None = None):
        run_dir = Path(run_dir)
        # final_ema wins where a run wrote both: best-on-val selects the most
        # overfit epoch, because near-duplicates leak across the by-image split.
        ckpt_path = next((run_dir / n for n in ("final_ema.pt", "best.pt")
                          if (run_dir / n).is_file()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No final_ema.pt or best.pt in {run_dir}.")

        self.run_dir = run_dir
        self.device = torch.device(device)
        # float32, not float16: DINOv3 ViT-L overflows in pure half precision and
        # returns NaN logits. DINOv2 tolerated fp16, which is why this was missed.
        self.dtype = dtype or torch.float32

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes: List[str] = ckpt["classes"]
        cfg = json.loads(json.dumps(ckpt["config"]))
        cfg["model"]["pretrained"] = False  # the checkpoint carries the weights

        self.image_size = int(cfg["data"]["image_size"])
        self.tta_scales = [float(s) for s in cfg["eval"]["tta_scales"]]
        self.head_type = cfg["model"]["head"]

        self.model = build_model(cfg, len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(dtype=self.dtype)
        del ckpt["model"]
        self.model.to(self.device)

        self.val_macro_f1 = float(ckpt.get("val_macro_f1", float("nan")))
        self.temperature, self.bands = self._load_calibration(run_dir)
        self.views = [_tta_transform(self.image_size, s, flip)
                       for s in self.tta_scales for flip in (False, True)]

    @staticmethod
    def _load_calibration(run_dir: Path) -> Tuple[float, List[dict]]:
        """Temperature + coverage/accuracy bands. Prefers calibration_T1.json —
        the T=0.381 in benchmark_metrics.json was fitted on a leaky by-image split
        and inflated mean confidence to 98.2% against 84.2% real accuracy."""
        for name in ("calibration_T1.json", "benchmark_metrics.json"):
            path = run_dir / name
            if path.is_file():
                bm = json.loads(path.read_text())
                bands = sorted(bm.get("coverage_accuracy", []),
                               key=lambda c: c["confidence_threshold"], reverse=True)
                return float(bm.get("temperature", 1.0)), bands
        raise FileNotFoundError(f"No calibration file in {run_dir}.")

    def _reliability(self, conf: float) -> str:
        for b in self.bands:
            if conf >= b["confidence_threshold"]:
                return (f"benchmark accuracy {b['accuracy'] * 100:.0f}% among the top "
                        f"{b['coverage'] * 100:.0f}% most-confident photos")
        if self.bands:
            low = self.bands[-1]
            return (f"below the {low['coverage'] * 100:.0f}%-coverage threshold "
                    f"({low['confidence_threshold']:.2f}) — treat as a weak guess")
        return "no calibration data"

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        logits = []
        for tf in self.views:
            x = tf(image=image)["image"].unsqueeze(0).to(self.device, dtype=self.dtype)
            logits.append(self.model(x)["logits"].float().cpu())
        mean_logits = torch.stack(logits).mean(0)
        probs = F.softmax(mean_logits / self.temperature, dim=1)[0].numpy()

        order = np.argsort(-probs)
        ranked = [{"key": self.classes[i],
                   "name": SPECIES_INFO[self.classes[i]][0],
                   "latin": SPECIES_INFO[self.classes[i]][1],
                   "prob": float(probs[i])} for i in order]

        return {
            "top": ranked[0]["key"],
            "top_name": ranked[0]["name"],
            "confidence": ranked[0]["prob"],
            "reliability": self._reliability(ranked[0]["prob"]),
            "n_views": len(self.views),
            "ranked": ranked,
        }
