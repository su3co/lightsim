# Contributing to lightsim

## Repository layout

```text
lightsim/
├── crates/                  Rust workspace
│   ├── lightsim-core/       simulation description, grid, materials, rasterizer, plan compiler, CPU engine, drivers, results, gradients, validation cases
│   ├── lightsim-cuda/       GPU backend: device memory, kernel launches, multi-GPU slabs (loads the committed PTX)
│   ├── lightsim-fem/        finite elements: meshing, heat, electrostatics, modes
│   ├── lightsim-cli/        the `lightsim` command (run, bench, validate, rank, launch, gds, info)
│   └── lightsim-py/         Python extension module `lightsim._core` (PyO3)
├── kernels/                 GPU kernels in Rust, compiled to PTX with cuda-oxide (standalone crate)
├── python/
│   ├── lightsim/            Python package: simulation API, mode solvers, S-parameters, EME, circuits, inverse design, FEM front end, GUI
│   └── tests/               Python test suite (pytest), with its reference data in fixtures/
├── docs/                    documentation pages (built into the website), site.json, landing.json, figures in img/
├── examples/                runnable examples: gdsfactory MMI S-parameters, Mie sphere (3-D), Mie cylinder (2-D)
├── benchmarks/              the scripts and result files behind every measured number (see benchmarks/README.md)
│   ├── speed/               FDTD step throughput against FDTDX, multi-node, feature scenes, coupler, metalens
│   ├── gradients/           time per inverse-design gradient, engine gradients
│   ├── two_d/               2-D update throughput and 2.5-D applications
│   ├── invdes/              time per gradient against Meep, FDTDX, ceviche, fdtd-z
│   ├── gplugins/            gdsfactory plugin swap test against Meep
│   └── scaling/             research prototypes for devices too large for one simulation
├── scripts/                 developer tools: kernel build, docs check, API reference, multi-node launcher, sync, Enzyme tests, resource monitor of benchmark runs, the comparison page
└── .github/                 CI workflows (Rust checks, wheels, release) and their helper scripts
```

The version is set once, in `Cargo.toml` (see [RELEASING.md](RELEASING.md)).

## Checks

What CI runs (`.github/workflows/`), and what to run before a pull request:

```bash
cargo clippy --locked -p lightsim-core -p lightsim-fem -p lightsim-cli -p lightsim-py --all-targets   # no warnings
cargo test --locked -p lightsim-core -p lightsim-fem                                                # CPU, validation cases included
maturin develop --release && pytest python/tests                                                     # Python suite on the CPU
python scripts/check_samples.py --no-run                                                             # docs lint, API reference fresh
```

The GPU checks need an NVIDIA GPU and are run by hand: `lightsim validate --device cuda:0`
(also with `LIGHTSIM_FUSED=0`, and on two GPUs with `--device cuda:0,1`),
`cargo test -p lightsim-cuda --release`, the Python suite with
`LIGHTSIM_DEVICE=cuda:0`, and `python scripts/check_samples.py` to run the
docs samples. [docs/validation.md](docs/validation.md) lists the physics cases.

After a change to the public Python API, regenerate the reference with
`python scripts/gen_api.py`. After a change to `kernels/`, rebuild the PTX with
`scripts/build_kernels.sh` (needs the cuda-oxide toolchain).
