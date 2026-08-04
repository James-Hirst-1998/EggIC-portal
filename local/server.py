"""Local development server — stands in for Vercel.

    python local/server.py [--port 8000]

Serves everything in ``public/`` and mirrors the two API routes that Vercel
provides in production, so the same frontend runs unchanged in both places.

The one deliberate difference: this server sends ``store: false``, so nothing you
test locally is written to the submissions volume. The deployed site stores every
submission. That asymmetry is the point of running locally.

Reads EGGIC_PREDICT_URL and EGGIC_API_KEY from the environment or from .env at
the repo root.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "public"
MAX_BODY = 40 * 1024 * 1024  # a phone photo, base64-inflated, with headroom

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml", ".webp": "image/webp", ".ico": "image/x-icon",
}


def load_env() -> tuple[str, str]:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    endpoint = os.environ.get("EGGIC_PREDICT_URL") or os.environ.get("EGGIC_ENDPOINT")
    key = os.environ.get("EGGIC_API_KEY")
    if not endpoint or not key:
        raise SystemExit(
            "Set EGGIC_PREDICT_URL and EGGIC_API_KEY (in the environment or in "
            f"{env_file}).\nEGGIC_PREDICT_URL is the predict URL printed by `modal deploy`."
        )
    return endpoint, key


def page_html(name: str) -> bytes:
    """A page with the logo inlined as a data URI.

    Inlining means the mark still renders when the file is opened straight off
    disk, which is how it silently went missing before — a bare /assets/ path only
    resolves when something is serving it. Vercel serves the path directly, so
    production does not need this.
    """
    html = (PUBLIC / name).read_text()
    logo = PUBLIC / "assets" / "shark_trust.png"
    if logo.is_file():
        uri = "data:image/png;base64," + base64.b64encode(logo.read_bytes()).decode()
        html = html.replace('src="/assets/shark_trust.png"', f'src="{uri}"')
    return html.encode()


class Handler(BaseHTTPRequestHandler):
    endpoint = ""
    key = ""

    def log_message(self, fmt, *args):
        pass  # the interesting log line is printed per prediction below

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, page_html("index.html"), CONTENT_TYPES[".html"])
            return
        if path in ("/about", "/about.html"):
            self._send(200, page_html("about.html"), CONTENT_TYPES[".html"])
            return

        # Everything else comes out of public/; resolve() plus the prefix check
        # keeps path traversal inside that directory.
        target = (PUBLIC / path.lstrip("/")).resolve()
        if (target.is_file() and target.is_relative_to(PUBLIC.resolve())
                and target.suffix.lower() in CONTENT_TYPES):
            self._send(200, target.read_bytes(), CONTENT_TYPES[target.suffix.lower()])
        else:
            self._json(404, {"error": f"no route {path}"})

    def do_POST(self) -> None:
        if self.path not in ("/api/predict", "/api/feedback"):
            self._json(404, {"error": f"no route {self.path}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": f"body must be 1..{MAX_BODY} bytes, got {length}"})
            return

        req = json.loads(self.rfile.read(length))

        if self.path == "/api/feedback":
            # Local runs never stored the photo, so there is nothing to attach
            # this to. Say so rather than pretending it saved.
            verdict = "correct" if req.get("correct") else f"wrong -> {req.get('user_species')}"
            print(f"  feedback: {verdict}  (not stored — local)", flush=True)
            self._json(200, {"stored": False, "reason": "local testing does not store photos"})
            return

        payload = json.dumps({
            "image": req.get("image"),
            "filename": req.get("filename"),
            "key": self.key,
            "store": False,   # local runs never write to the submissions volume
        }).encode()

        request = urllib.request.Request(
            self.endpoint, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            self._json(exc.code, {"error": exc.read().decode()[:500]})
            return
        except urllib.error.URLError as exc:
            self._json(502, {"error": f"could not reach Modal: {exc.reason}"})
            return

        result = json.loads(body)
        print(f"  {result['top_name']:<24} {result['confidence'] * 100:5.1f}%  "
              f"{result['latency_ms']}ms  ({result['model_version']})", flush=True)
        self._send(200, body, "application/json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    Handler.endpoint, Handler.key = load_env()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Viewer at {url}\nProxying to {Handler.endpoint}\n"
          "Local runs are not stored. The first prediction after idle waits on a cold start.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
