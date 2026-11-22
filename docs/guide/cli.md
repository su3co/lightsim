# Command line

After this page you can run a simulation from a JSON file with the `lightsim`
program, read its results file, benchmark a machine, and start a run that
spans several machines.

The program is built as described in [Install](install.md). It takes a
command followed by arguments; options are written `--name value`.

| Command | Purpose |
|---|---|
| `lightsim info` | Version, CPU threads and GPUs |
| `lightsim validate` | Run the physics validation suite |
| `lightsim run SPEC.json` | Run a simulation |
| `lightsim bench` | Measure throughput on a synthetic problem |
| `lightsim rank SPEC.json` | Run one part of a multi-machine simulation |
| `lightsim launch SPEC.json` | Start all parts of a multi-machine simulation over ssh |

Devices are named as in Python: `cpu`, `cuda` (the first GPU), `cuda:1`,
`cuda:0,1` (several GPUs) and `cuda:all`. Without `--device`, the first GPU
is used if the program was built with GPU support and one is present, and
the CPU otherwise.

## Running a simulation

```bash
lightsim run SPEC.json --device cuda:0 --out results.json --progress 500
```

| Option | Meaning |
|---|---|
| `--device` | Where to run |
| `--out FILE` | Write the results as JSON. Without it nothing is saved |
| `--progress N` | Print a line every N steps |
| `--dimensions 2` | Run the file as a 2-D simulation (see [2-D, 2.5-D and 3-D](dimensions.md)); `3` runs a 2-D file in 3-D |
| `--invariant-axis x`, `y` or `z` | The axis along which nothing varies; implies `--dimensions 2` |
| `--polarization te`, `tm` or `both` | The field set of a 2-D simulation: `te` has the electric field in the plane, `tm` along the invariant axis |
| `--kernels 3d` | Run a 2-D simulation through the general 3-D update, to cross-check a result |

The last four options override the `dimensions`, `invariant_axis`,
`polarization` and `force_3d_kernels` keys of the file.

The simulation is described by a JSON file. It has the same content as a
Python `Simulation`: the file is what `Simulation.to_json()` writes, so a
simulation can be built in Python and run by the program, or written by hand
or by another tool. The complete format is in the
[JSON reference](../reference/json-spec.md).

A small file, a dipole in an empty box with two monitors:

```json
{
  "name": "dipole_box",
  "region": {"min": [0, 0, 0], "max": [3e-6, 1.5e-6, 1.5e-6]},
  "mesh": {"type": "uniform", "dx": [50e-9, 50e-9, 50e-9]},
  "boundaries": {
    "x": [{"type": "pml"}, {"type": "pml"}],
    "y": [{"type": "pml"}, {"type": "pml"}],
    "z": [{"type": "pml"}, {"type": "pml"}]
  },
  "run_time": 40e-15,
  "sources": [
    {"type": "point_dipole", "name": "dipole",
     "position": [1.5e-6, 0.75e-6, 0.75e-6], "polarization": "Ez",
     "signal": {"type": "gaussian_pulse", "freq0": 3e14, "fwidth": 1e14}}
  ],
  "monitors": [
    {"type": "flux_box", "name": "power",
     "min": [1.0e-6, 0.25e-6, 0.25e-6], "max": [2.0e-6, 1.25e-6, 1.25e-6], "freqs": [3e14]},
    {"type": "probe", "name": "probe", "position": [2.5e-6, 0.75e-6, 0.75e-6], "components": ["Ez"]}
  ]
}
```

Running it on the CPU prints the progress and a summary to the terminal:

```text
  step 200/420 (0.4 s)
  step 400/420 (0.7 s)
cpu (10 threads): 244944 cells x 420 steps in 0.744 s = 138.2 Mcells/s
```

The worked example does the same from Python: it writes the file, calls the
program and reads the results. The first block builds the same simulation
with the Python builders.

```python
import json
import shutil
import subprocess

import lightsim as ls
from lightsim.spec import um

sim = ls.Simulation(min=[0, 0, 0], max=[3 * um, 1.5 * um, 1.5 * um],
                    mesh=ls.UniformMesh(50e-9), run_time=40e-15, name="dipole_box")
sim.add_source(ls.PointDipole([1.5 * um, 0.75 * um, 0.75 * um], "Ez",
                              ls.GaussianPulse(3e14, 1e14), name="dipole"))
sim.add_monitor(ls.FluxBoxMonitor("power", [1.0 * um, 0.25 * um, 0.25 * um],
                                  [2.0 * um, 1.25 * um, 1.25 * um], [3e14]))
sim.add_monitor(ls.ProbeMonitor("probe", [2.5 * um, 0.75 * um, 0.75 * um], components=("Ez",)))

with open("dipole_box.json", "w") as f:
    f.write(sim.to_json())

program = shutil.which("lightsim")
subprocess.run([program, "run", "dipole_box.json", "--device", "cpu", "--out", "results.json"], check=True)

with open("results.json") as f:
    results = json.load(f)
print(sorted(results))
print(results["stats"]["cells"], results["stats"]["steps"], results["stats"]["stopped_early"])
print(len(results["flux"]["power"]), results["time"]["probe"]["fields"][0]["dims"])
```

```text
['dft', 'dt', 'energy', 'far_field', 'flux', 'modes', 'nsteps', 'source_spectra', 'sources', 'stats', 'time']
244944 420 False
1 [420, 1, 1, 1]
```

## The results file

The file holds the same data as the Python `Results` object, as nested JSON
objects with flat arrays. The [JSON reference](../reference/json-spec.md)
lists every field. One thing differs from Python: arrays are flat lists with
a separate `dims` entry. An array with dimensions `[n0, nx, ny, nz]` stores
the element `(f, i, j, k)` at position `((f * nx + i) * ny + j) * nz + k`.

The file is plain text and grows with the monitors. A volume field monitor
with several frequencies can produce gigabytes; record planes where they
suffice.

## Looking at a spec and its results

The graphical interface of the Python package opens a spec file and a
results file written by `lightsim run --out` in a browser: cuts of the
rasterised structure with the sources and monitors, a run view, and plots of
every monitor (see [Graphical interface](gui.md)).

```bash
python -m lightsim.gui SPEC.json --results results.json
```

The `lightsim` program has no command for this: it is built from Rust
alone, and the interface is part of the Python package.

## Validation

```bash
lightsim validate --device cuda:0
lightsim validate --device cpu --case dipole
```

Runs the validation cases on the chosen device and prints, for each check,
the measured value, the reference and the tolerance. `--case TEXT` restricts
the run to the cases whose name contains the text. The exit status is 1 if a
case fails. See [Validation](../validation.md).

## Benchmark

```bash
lightsim bench --device cuda:0 --n 400,400,400 --steps 200
```

Runs a box with a dipole and absorbing boundaries for a fixed number of steps
and prints the throughput in million cell updates per second.

| Option | Meaning | Default |
|---|---|---|
| `--n NX,NY,NZ` | Grid size in cells, absorbing layers included | `256,256,256` |
| `--n NU,NV` | Two numbers: a 2-D grid, with an electric dipole per field set in use | |
| `--invariant-axis`, `--polarization`, `--kernels 3d` | For a 2-D grid, as for `run`; `--kernels 3d` times the same one-cell grid on the 3-D update | `z`, `both` |
| `--steps N` | Number of time steps | 200 |
| `--scene NAME` | `vacuum`, `lorentz` (a dispersive medium everywhere), `bloch` (Bloch boundaries along x and y; along the second in-plane axis in 2-D) or `tensor` (a rotated crystal everywhere, 3-D only) | `vacuum` |
| `--dispersive` | Add a gold sphere | off |
| `--json FILE` | Also write the numbers as JSON | none |
| `--membw --gpu I` | Measure the memory copy bandwidth of GPU I instead | off |

On the CPU of a 10-core laptop:

```text
bench Cpu grid [96, 96, 96] = 0.9 Mcells, 50 steps: 0.325 s (0.01 s setup) -> 136.0 Mcells/s (~14 GB/s effective) [cpu (10 threads)]
```

The CPU engine uses all cores. Set the environment variable
`RAYON_NUM_THREADS` to limit the number of threads.

FDTD throughput is bounded by how fast memory can be read and written.
`lightsim bench --membw --gpu 0` measures that limit for a GPU; on an
RTX PRO 6000 it printed:

```text
cuda:0 copy bandwidth: 1488.1 GB/s
```

## Several machines

A simulation can be split along x over several machines, one part ("rank")
per machine, each using one or more of its GPUs. Neighbouring ranks exchange
the fields on their shared faces over TCP at every step. Rank 0 collects the
results and writes the output file.

```bash
lightsim launch SPEC.json --hosts node0,node1,node2 --out results.json
```

`launch` starts `lightsim rank` on every host over ssh and waits for all of
them. The same is done by the script `examples/multi_node.sh` in the
repository, and can be done by hand:

```bash
lightsim rank SPEC.json --hosts node0,node1,node2 --rank 0 --out results.json
lightsim rank SPEC.json --hosts node0,node1,node2 --rank 1
lightsim rank SPEC.json --hosts node0,node1,node2 --rank 2
```

| Option of `rank` | Meaning | Default |
|---|---|---|
| `--hosts A,B,...` | Addresses at which the ranks reach each other, rank 0 first | required |
| `--rank R` | Which rank this process is | required |
| `--gpus I,J` | GPUs this rank uses | `0` |
| `--cpu` | Use the CPU engine instead of GPUs | off |
| `--weights W0,W1,...` | Relative size of each rank's part, for unequal machines | equal |
| `--port P` | Base TCP port; rank R listens on P + R | 47100 |
| `--chain-addrs A1,A2,...` | Address rank R - 1 uses to reach rank R for the field exchange, when neighbours are joined by a dedicated link | from `--hosts` |
| `--steps N` | Override the number of steps | from the file |
| `--out FILE`, `--json FILE` | Results and a short summary, written by rank 0 | none |
| `--progress N` | Progress lines from rank 0 | off |

`launch` accepts the same options and passes them on, plus `--addrs` (the
addresses to pass as `--hosts` to the ranks, when the ssh names differ from
them) and `--bin` (the path of the program on the remote machines).

Requirements:

- The program and the JSON file must exist at the same path on every
  machine, and `launch` needs ssh access without a password prompt.
- The base port and the following ports, one per rank, must be reachable
  between the machines.
- Ranks wait up to two minutes for each other at start-up. There is no
  recovery if a rank fails during the run.

Two ranks can be tried on one machine with the CPU engine, in two terminals:

```bash
lightsim rank dipole_box.json --hosts localhost,localhost --rank 1 --cpu
lightsim rank dipole_box.json --hosts localhost,localhost --rank 0 --cpu --out results.json
```

```text
[rank 0] slab x=[0,42) over 1 GPU(s) on localhost
[rank 0] links up (lo=false, hi=true)
[rank 0] 244944 cells x 420 steps in 1.239 s = 83.0 Mcells/s (2 ranks x 1 GPU(s))
```

The messages say "GPU(s)" also when the CPU engine is used. The split run
writes exactly the same flux and probe values as the single run.
[GPUs and scaling](gpu.md) discusses how the domain is divided and what to
expect.

## Using the solver from Rust

The program and the Python package are thin layers over two Rust crates in
the repository, which can be used directly:

| Crate | Contents |
|---|---|
| `lightsim-core` | The simulation description, meshing, the CPU engine and the results. `run_cpu(&sim)` runs a simulation on the CPU |
| `lightsim-cuda` | The GPU engine. `run_gpu(&sim, gpu_index)` runs on one GPU and `multi::run_multi_gpu(&sim, &gpus, &options)` on several |

A simulation is read from the same JSON:

```rust
use lightsim_core::{Simulation, run_cpu};

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("dipole_box.json")?;
    let sim = Simulation::from_json(&json)?;
    let results = run_cpu(&sim)?;
    println!("{} steps, flux {:?}", results.stats.steps, results.flux["power"]);
    Ok(())
}
```

One capability exists only at this level: stepping a simulation backwards in
time (`lightsim_core::reverse`), which reconstructs earlier fields from later
ones. It is not available from Python or from the command line. It supports
linear, non-dispersive media without off-diagonal terms.
