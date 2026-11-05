"""What the scripts here share: settings from .env, and Supabase calls with the secret key.

The secret key bypasses row-level security. It belongs in .env (gitignored) and the
eggic-supabase Modal secret, nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EGGIC_DATA", ROOT.parent / "EggIC_data"))

SPECIES = ["BlondeRay", "CuckooRay", "FlapperSkate", "Nursehound", "SmallEyedRay",
           "SmallSpottedCatshark", "SpottedRay", "ThornbackRay", "UndulateRay"]


def env(name: str) -> str:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} in the environment or in {env_file}")
    return value


def call(method: str, path: str, **kwargs) -> requests.Response:
    """One Supabase REST / Storage / Auth admin call as the service role."""
    headers = {"apikey": env("SUPABASE_SECRET_KEY"), **kwargs.pop("headers", {})}
    resp = requests.request(method, env("SUPABASE_URL").rstrip("/") + path,
                            headers=headers, timeout=60, **kwargs)
    if not resp.ok:
        raise SystemExit(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
    return resp


def upload(bucket: str, path: str, data: bytes) -> None:
    call("POST", f"/storage/v1/object/{bucket}/{path}", data=data,
         headers={"Content-Type": "image/jpeg", "x-upsert": "true"})


def upsert(table: str, rows: list[dict]) -> None:
    call("POST", f"/rest/v1/{table}", json=rows,
         headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
