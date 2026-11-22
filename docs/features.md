# Features: lightsim, Lumerical FDTD and FDTDX

| | lightsim | Lumerical FDTD | FDTDX |
|---|---|---|---|
| 3-D Yee FDTD, f32 | ✅ | ✅ | ✅ |
| 2-D FDTD: TE (E in the plane), TM (E along the invariant axis) or both, three components updated per polarization | ✅ (any invariant axis; CPU, GPU, multi-GPU) | ✅ (xy plane) | one-cell 3-D grid, six components |
| Non-uniform rectilinear mesh (auto mesh + override regions) | ✅ | ✅ | ✅ |
| Sub-pixel (anisotropic) smoothing of dielectric interfaces | ✅ | ✅ (conformal) | ✅ |
| Dielectric, diagonal-anisotropic, conductive (electric and magnetic), magnetic, PEC media | ✅ | ✅ | ✅ |
| Dispersive media: Lorentz / Drude / Debye (ADE), per-axis poles; dense or sparse pole state | ✅ | ✅ | ✅ |
| Oriented dispersive poles χ(ω) u uᵀ (rotated dispersive crystals, monoclinic media) | ✅ | ✅ | ✅ |
| Material library (Si, SiO2, Si3N4, Au, Ag Rakić fits, …) | ✅ | ✅ | partial |
| Sheet materials given by a surface conductivity: graphene (Kubo), 2-D conductors (`lightsim.sheets`) | ✅ (as a one-cell dispersive layer) | ✅ (zero-thickness surface) | ❌ |
| Sources: E/M dipoles, plane wave (incl. oblique s- and p-polarised, narrow-band), Gaussian beam (focused), eigenmode, TFSF (1-D auxiliary grid, leakage 1e-7; dispersive backgrounds via ADE on the auxiliary line) | ✅ | ✅ | ✅ |
| Spectra over a band at one angle of incidence | ✅ (`lightsim.sweep_angle`: one narrow-band run per frequency, spread over the GPUs) | ✅ (BFAST: one broadband run) | ❌ |
| Boundaries: CPML (κ, α), periodic, PEC, PMC / symmetry planes | ✅ | ✅ | ✅ |
| Automatic mirror-symmetry reduction with unfolded results (`run_symmetric`) | ✅ | ✅ | ✅ |
| Exact time reversal of the leapfrog (PML replayed from a recording) | ✅ (CPU and GPU) | — | ✅ |
| Bloch-periodic with k ≠ 0 (complex fields, any axis) | ✅ (1 GPU, multi-GPU, multi-node) | ✅ | ✅ |
| Monitors: field-time, probe, DFT field (apodization, stride), flux plane, closed flux box, mode expansion (S-params), energy | ✅ | ✅ | ✅ |
| Far-field projection (surface equivalence, U(θ, φ)) | ✅ | ✅ | ✅ |
| Eigenmode solver (full-vectorial FD on the FDTD Yee grid, complex ε for lossy/dispersive guides) | ✅ | ✅ (FDE) | ✅ |
| gdsfactory / GDS + layer-stack import, port-driven S-parameter sweeps | ✅ | ✅ | ✅ |
| Field-decay auto shutoff | ✅ | ✅ | ✅ |
| Imported / graded index grids, silicon carrier-plasma (Soref–Bennett) perturbations (`IndexGrid`, `lightsim.eo`) | ✅ | ✅ | ❌ |
| Material library (Sellmeier: SiO2, Si3N4, Si, LiNbO3, Al2O3, GaAs, Ge) and passive Lorentz/Drude fitting (`lightsim.materials`) | ✅ | ✅ | partial |
| Resonance analysis: harmonic inversion, Q, Lorentzian fits (`lightsim.resonance`) | ✅ | ✅ | ❌ |
| Grating / diffraction-order efficiencies (`lightsim.gratings`) | ✅ | ✅ | ❌ |
| 2.5-D effective-index (varFDTD-style) simulations and S-parameters on the 2-D solver (`lightsim.eim`) | ✅ (plain EIM) | ✅ (variational) | ❌ |
| Multi-GPU in one node | ✅ (P2P, ~2× on 2 GPUs) | ✅ | ✅ |
| Multi-node | ✅ (TCP, 96 % efficiency on 3 nodes) | ✅ (MPI, CPU) | ✅ (jax.distributed) |
| Double precision | ✅ on the CPU (`f64` build: 0.5–0.7× the speed, twice the memory); the GPU kernels are single precision | ✅ | ✅ |
| Graphical interface | ✅ in the browser (`lightsim.gui`): the set-up as the solver rasterises it, a run with progress and a live field slice, results (spectra, S-parameters, field maps, mode profiles, far field); no geometry editing | ✅ desktop CAD application with a script console | ❌ |
| Python API + CLI + Rust API | ✅ | Python/lsf | Python |
| Adjoint gradients / inverse design (mode-power FOM, dF/dε per Yee cell) | ✅ (Python, 2 runs) | ✅ (lumopt) | ✅ (autodiff) |
| Shape optimisation: polygons with movable vertices and width profiles, with adjoint gradients (`lightsim.invdes.ShapeDesign`) | ✅ (shapes drawn on the design grid; gradient within 0.1 % with the exact `reverse` engine, within 0.7 % in 2.5-D and 4 % in 3-D with the faster adjoint engines) | ✅ (lumopt, boundary integral) | ❌ |
| Gradients inside the engine: persistent S-matrix session (design sub-block updates, adjoint product on the GPU) reverse mode through the time loop for time-domain objectives, and exact S-matrix gradients through it (`engine="reverse"`: 1e-4 against finite differences, about four times the cost of the adjoint) (`lightsim.autodiff`) | ✅ (CPU and GPU) | — | ✅ (autodiff) |
| Nonlinear χ², χ³ (instantaneous, per axis; implicit per-cell Newton solve) | ✅ | ✅ | ❌ |
| Raman response of the χ³ nonlinearity (damped-oscillator memory, per axis) | ✅ | ✅ (Chi3 Raman Kerr) | ❌ |
| Gain media | ✅ (four-level rate equations with pumping, saturation, exact atom count) | ✅ (four-level two-electron model, current-driven gain) | linear only (negative loss) |
| Magneto-optic (gyrotropic) media: Lorentz and Drude poles in a static bias along any direction; Faraday rotation | ✅ | ✅ (complex tensor through a matrix transformation) | partial (antisymmetric conductivity) |
| Anisotropy whose orientation varies from cell to cell (liquid-crystal director fields) | ✅ | ✅ | ❌ |
| Off-diagonal ε and σ tensors (rotated, lossy anisotropic crystals) | ✅ (1 GPU, multi-GPU, multi-node) | ✅ | ✅ |

## In Lumerical FDTD, not in lightsim

The table lists what lightsim has. These Lumerical FDTD capabilities have no
counterpart in lightsim:

- **Lumerical's own gain models**: the four-level two-electron model and
  current-driven gain. lightsim's gain medium is a four-level rate-equation
  model with a pump rate; it gives small-signal gain and saturation, not
  Lumerical's formulation.
- **Complex permittivity tensors given directly.** lightsim's magneto-optic
  media are gyrotropic poles in a static bias (a physical model that yields
  the antisymmetric tensor), not an arbitrary Hermitian tensor.
- **Broadband oblique incidence in one run** (BFAST). `lightsim.sweep_angle`
  gives the same spectra from one run per frequency.
- **Zero-thickness sheets.** lightsim represents a sheet by a layer one cell
  thick.
- **User-defined material plugins.**
- **A CAD application.** lightsim's interface ([`lightsim.gui`](guide/gui.md),
  in the browser) shows the set-up, a run and the results. It does not edit
  geometry and has no object library or script console: structures come from
  Python and gdsfactory.

## What Lumerical's GPU solver leaves to the CPU

The table compares lightsim with Lumerical FDTD as a whole. Lumerical's GPU
solver covers less than its CPU solver. According to Ansys
([Getting started with running FDTD on GPU](https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU),
read on 2026-11-22), on a GPU Lumerical FDTD:

- runs 3-D simulations only;
- has no TFSF source and no BFAST source;
- does not support its sheet (2-D conductivity) materials, except a PEC
  sheet, its other advanced permittivity models (the family that holds the
  nonlinear, gain and magneto-optic models), except those with a temperature
  grid, or materials from its plug-in framework;
- records no movies, switches off time apodization and spectral averaging in
  frequency-domain monitors, and warns that time monitors slow the run;
- has no checkpoints.

In lightsim every item of the table above runs on the GPU unless its row says
otherwise (double precision is CPU-only): 2-D simulations, TFSF sources,
χ² and χ³ media with the Raman response, gain media, magneto-optic media,
director fields and apodized DFT monitors each have a validation case that
passes on the RTX PRO 6000 and on the GB10 ([Validation](validation.md)),
and sheet layers a test of the Python suite that passes on the RTX PRO 6000.
On a GPU, then, lightsim covers more of this
table than Lumerical does; on a CPU Lumerical covers the list above this
section, which lightsim does not have.

Module notes with their validation numbers: [material library](modules/materials.md),
[sheet materials](modules/sheets.md), [shape optimisation](modules/shapes.md),
[index grids](modules/index_grid.md), [gratings](modules/gratings.md),
[resonances](modules/resonance.md) and [effective index](modules/eim.md).
