#!/usr/bin/env bash
# Run a lightsim spec across several machines, one rank per machine (one or
# more GPUs each). Halos travel over TCP between neighbouring ranks; control
# messages and results go to rank 0.
#
# Usage: HOSTS=h0,h1,h2 [CHAIN=a1,a2] examples/multi_node.sh SPEC.json [--steps 200]
#   HOSTS  addresses at which each rank is reachable (rank 0 first)
#   CHAIN  optional per-link addresses: CHAIN[r-1] is rank r's address on its
#          direct link to rank r-1 (e.g. point-to-point 200 GbE cables)
#   LIGHTSIM_BIN  path of the lightsim binary on the remote machines (default: lightsim)
set -euo pipefail
SPEC=$1; shift
IFS=, read -ra H <<< "${HOSTS:?set HOSTS=h0,h1,...}"
BIN=${LIGHTSIM_BIN:-lightsim}
CHAIN_ARG=${CHAIN:+--chain-addrs $CHAIN}
for r in "${!H[@]}"; do
  ssh "${H[$r]}" "$BIN rank $SPEC --hosts $HOSTS $CHAIN_ARG --rank $r $*" &
done
wait
