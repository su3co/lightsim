"""Graphical interface: a local web application to inspect a simulation
before running it, watch a run and read the results.

    import lightsim as ls
    url = sim.view()                     # or lightsim.gui.view(sim)
    url = lightsim.gui.view(sim, results=res)
    url = lightsim.gui.view(res)         # results alone

From a shell, for a spec written by ``Simulation.to_json()``:

    python -m lightsim.gui spec.json [--results results.json] [--port 8765]

The server listens on 127.0.0.1 on a free port and prints the URL. It
needs nothing beyond the standard library and numpy, and the page loads
nothing from the network.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser

from .app import Session
from .server import Server

__all__ = ["view", "serve", "stop", "wait"]

_servers: list[Server] = []


def _display_available() -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def serve(sim=None, results=None, *, host: str = "127.0.0.1", port: int = 0, ports=None, title: str | None = None) -> Server:
    """Start the interface in a background thread and return the
    :class:`Server` (``.url``, ``.stop()``).

    ``sim`` is a ``Simulation``, a spec dict or a JSON string; ``results`` a
    ``Results`` (or its ``raw`` dict). A ``Results`` passed as ``sim`` is
    taken as the results. ``ports`` is an optional list of port dicts (as
    ``lightsim.gds.port_info`` returns) drawn on the cuts. ``host`` other
    than a loopback address makes the server reachable from other machines;
    the URL then carries a random access token.
    """
    if results is None and sim is not None and hasattr(sim, "raw") and not hasattr(sim, "to_dict"):
        sim, results = None, sim
    srv = Server(Session(sim, results, ports=ports, title=title), host=host, port=port).start()
    _servers.append(srv)
    return srv


def view(sim=None, results=None, *, host: str = "127.0.0.1", port: int = 0, open_browser: bool | None = None, block: bool = False, ports=None, title: str | None = None) -> str:
    """Open the interface for a simulation and/or its results.

    Starts a server (see :func:`serve`), prints its URL and returns it.
    ``open_browser`` opens the URL in the default browser; by default it
    does so when a display is available and the session is not over SSH.
    ``block=True`` serves until interrupted (Ctrl-C), as a script that
    ends would otherwise stop the server with it.
    """
    srv = serve(sim, results, host=host, port=port, ports=ports, title=title)
    print(f"lightsim interface: {srv.url}", flush=True)
    if open_browser or (open_browser is None and _display_available()):
        threading.Thread(target=webbrowser.open, args=(srv.url,), daemon=True).start()
    if block:
        wait()
    return srv.url


def wait():
    """Serve until interrupted (Ctrl-C), then stop every server."""
    try:
        while any(s.thread is not None and s.thread.is_alive() for s in _servers):
            for s in list(_servers):
                if s.thread is not None:
                    s.thread.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop()


def stop():
    """Stop every server started by :func:`view` or :func:`serve`."""
    while _servers:
        _servers.pop().stop()
