"""``python -m lightsim.gui SPEC.json [--results RESULTS.json]``: serve the
interface for a simulation spec (and the results file of ``lightsim run
--out``) until interrupted."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m lightsim.gui", description="Graphical interface for a lightsim simulation: set-up, run, results.")
    p.add_argument("spec", nargs="?", help="simulation spec (JSON, as written by Simulation.to_json())")
    p.add_argument("--results", help="results file written by `lightsim run --out`")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1; another address adds an access token to the URL)")
    p.add_argument("--port", type=int, default=0, help="port (default: a free one)")
    p.add_argument("--open", action="store_true", help="open the URL in the default browser")
    a = p.parse_args(argv)
    if not a.spec and not a.results:
        p.error("give a spec, a results file or both")
    from . import Server, Session
    from .app import results_from_cli_json

    spec = None
    if a.spec:
        with open(a.spec) as f:
            spec = json.load(f)
    results = None
    if a.results:
        with open(a.results) as f:
            results = results_from_cli_json(json.load(f))
    srv = Server(Session(spec, results), host=a.host, port=a.port)
    print(f"lightsim interface: {srv.url}", flush=True)
    print("press Ctrl-C to stop", flush=True)
    if a.open:
        import webbrowser

        webbrowser.open(srv.url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        run = srv.session.run
        if run.running():
            run.stop()
            run.join(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
