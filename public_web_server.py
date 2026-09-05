"""Serve the standalone webpage and XGBoost prediction API from one origin."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from model_api_server import MODEL_DIR, predict_payload


ROOT = Path(__file__).resolve().parent
HTML_CANDIDATES = [
    ROOT / "outputs" / "web" / "ai_risk_agent_route_planning_3d_webpage.html",
    ROOT / "ai_risk_agent_route_planning_3d_webpage.html",
]
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8088"))


def html_path() -> Path | None:
    for candidate in HTML_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


class PublicWebHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, body: dict) -> None:
        self._send_bytes(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self._send_json(200, {})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            path_obj = html_path()
            if path_obj is None:
                self._send_json(404, {"ok": False, "error": "html not found"})
                return
            self._send_bytes(200, path_obj.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send_json(200, {"ok": True, "model_dir": str(MODEL_DIR), "html": str(html_path())})
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/predict":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = predict_payload(payload)
            self._send_json(200, {"ok": True, **result})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PublicWebHandler)
    print(f"Public web server running at http://{HOST}:{PORT}")
    print(f"Health check: http://{HOST}:{PORT}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
