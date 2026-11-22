"""A small HTTP server for the interface (standard library only).

The page and its scripts are served from the package's ``static`` folder;
``/api/...`` answers JSON computed by :class:`lightsim.gui.app.Session`;
``/api/run/events`` streams the state of a run as server-sent events.

By default the server listens on 127.0.0.1 and accepts only requests whose
``Host`` header names the loopback interface (so that another web site
cannot reach it through a rebound DNS name). With another ``host`` every
request must carry a random token: the URL printed at start-up holds it,
the page keeps it in a cookie.
"""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import secrets
import threading
import time
import traceback
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .app import Session

STATIC = Path(__file__).resolve().parent / "static"
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def needs_token(host: str) -> bool:
    """A server on any address other than a loopback one needs a token."""
    return not is_loopback(host)


def _host_name(h: str) -> str:
    """The host part of a Host header (without the port)."""
    if h.startswith("["):
        return h.split("]")[0] + "]"
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


class _JSON(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _dumps(obj) -> bytes:
    # NaN and inf are not JSON: send them as null
    def clean(x):
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x

    return json.dumps(clean(json.loads(json.dumps(obj, cls=_JSON, allow_nan=True))), allow_nan=False).encode()


class Server:
    """The interface for one session. ``start()`` serves in a background
    thread; ``stop()`` shuts the server down and stops a run in progress."""

    def __init__(self, session: Session, host: str = "127.0.0.1", port: int = 0, token: str | None = None):
        self.session = session
        self.host = host
        self.loopback = is_loopback(host)
        self.token = token if token is not None else (secrets.token_urlsafe(18) if needs_token(host) else None)
        handler = type("Handler", (_Handler,), {"app": self})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = None
        self._closing = threading.Event()

    @property
    def url(self) -> str:
        h = self.host
        if h in ("0.0.0.0", "::", ""):
            shown = "127.0.0.1" if h != "::" else "[::1]"
        else:
            shown = f"[{h}]" if ":" in h and not h.startswith("[") else h
        u = f"http://{shown}:{self.port}/"
        if self.token:
            u += f"?token={self.token}"
        return u

    def start(self) -> "Server":
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2}, name="lightsim-gui", daemon=True)
        self.thread.start()
        return self

    def serve_forever(self):
        try:
            self.httpd.serve_forever(poll_interval=0.2)
        finally:
            self.httpd.server_close()

    def stop(self, timeout: float = 5.0):
        self._closing.set()
        run = self.session.run
        if run.running():
            run.stop()
            run.join(timeout)
        if self.thread is not None:
            self.httpd.shutdown()
            self.thread.join(timeout)
            self.thread = None
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class _Handler(BaseHTTPRequestHandler):
    app: Server
    server_version = "lightsim-gui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    # -- access rules ---------------------------------------------------------
    def _allowed(self, query) -> bool:
        app = self.app
        if app.loopback:
            host = _host_name(self.headers.get("Host") or "")
            if host and not is_loopback(host):
                return False
        if app.token is None:
            return True
        if query.get("token", [None])[0] == app.token:
            return True
        c = cookies.SimpleCookie(self.headers.get("Cookie") or "")
        if "lightsim_token" in c and secrets.compare_digest(c["lightsim_token"].value, app.token):
            return True
        if secrets.compare_digest(self.headers.get("X-Lightsim-Token") or "", app.token):
            return True
        return False

    def _send(self, code: int, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, _dumps(obj), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            raise ValueError("request too large")
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw or b"{}")

    # -- dispatch -------------------------------------------------------------
    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if not self._allowed(q):
            self._send(403, b"forbidden: open the URL printed when the server started (it holds the access token)\n", "text/plain; charset=utf-8")
            return
        try:
            if url.path.startswith("/api/"):
                if method == "POST" and self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self._error(415, "POST bodies must be application/json")
                    return
                self._api(method, url.path[5:], q)
            elif method == "GET":
                self._static(url.path, q)
            else:
                self._error(405, "method not allowed")
        except (KeyError, IndexError) as e:
            self._error(404, f"not found: {e}")
        except (ValueError, RuntimeError, TypeError) as e:
            self._error(400, str(e))
        except BrokenPipeError:
            pass
        except Exception as e:  # pragma: no cover - reported to the page
            traceback.print_exc()
            self._error(500, f"{type(e).__name__}: {e}")

    def _static(self, path, q):
        if path in ("", "/"):
            path = "/index.html"
        p = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in p.parents or not p.is_file():
            self._send(404, b"not found\n", "text/plain")
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        extra = {}
        if self.app.token and q.get("token", [None])[0] == self.app.token:
            extra["Set-Cookie"] = f"lightsim_token={self.app.token}; Path=/; HttpOnly; SameSite=Strict"
        self._send(200, p.read_bytes(), ctype, extra)

    def _api(self, method, route, q):
        s = self.app.session
        arg = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
        if route == "info":
            return self._json(s.info())
        if route == "cut":
            return self._json(s.cut(int(arg("axis", 2)), int(arg("index", 0)), arg("q", "n"), int(arg("side", 900))))
        if route == "outline":
            return self._json({"segments": s.outline_segments(int(arg("axis", 2)), int(arg("index", 0)))})
        if route == "results":
            return self._json(s.results_index())
        if route.startswith("results/"):
            if s.results is None:
                raise KeyError("no results")
            kind = route.split("/", 1)[1]
            name = arg("name")
            if kind == "flux":
                return self._json(s.flux_data(name))
            if kind == "mode":
                return self._json(s.mode_data(name))
            if kind == "profiles":
                return self._json(s.mode_profiles(name))
            if kind == "map":
                ax = arg("axis")
                ix = arg("index")
                return self._json(s.dft_map(name, arg("comp"), int(arg("f", 0)), arg("part", "abs"), None if ax in (None, "") else int(ax), None if ix in (None, "") else int(ix)))
            if kind == "time":
                return self._json(s.time_data(name, arg("comp")))
            if kind == "energy":
                return self._json(s.energy_data(name))
            if kind == "far":
                return self._json(s.far_data(name, int(arg("f", 0))))
            raise KeyError(kind)
        if route == "run/state":
            return self._json(s.run.snapshot())
        if route == "run/slice":
            sl = s.run.get_slice()
            return self._json(sl or {})
        if route == "run/events":
            return self._events()
        if method == "POST":
            body = self._body()
            if route == "run/start":
                if not s.has_spec:
                    raise ValueError("these results came without a simulation: nothing to run")
                every = min(max(float(body.get("every", 0.3)), 0.02), 5.0)
                return self._json(s.run.start(s, str(body.get("device", "cpu")), str(body.get("comp", "Ey")), int(body.get("axis", 2)), int(body.get("index", 0)), every))
            if route == "run/stop":
                return self._json(s.run.stop())
            if route == "run/plane":
                return self._json(s.run.set_plane(str(body.get("comp", "Ey")), int(body.get("axis", 2)), int(body.get("index", 0))))
        raise KeyError(route)

    def _events(self):
        """Server-sent events: the run state whenever it changes (at most
        five times a second), until the run ends or the client goes away."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        run = self.app.session.run
        last = -1
        idle_after_end = 0
        try:
            while not self.app._closing.is_set():
                snap = run.snapshot()
                if snap["version"] != last:
                    last = snap["version"]
                    self.wfile.write(b"data: " + _dumps(snap) + b"\n\n")
                    self.wfile.flush()
                    idle_after_end = 0
                elif not run.running():
                    idle_after_end += 1
                    if idle_after_end > 10:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        idle_after_end = 0
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass
