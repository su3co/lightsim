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
| Python API + CLI + Rust API | ✅ | Python/lsf | Python |
| Adjoint gradients / inverse design (mode-power FOM, dF/dε per Yee cell) | ✅ (Python, 2 runs) | ✅ (lumopt) | ✅ (autodiff) |
| Shape optimisation: polygons with movable vertices and width profiles, with adjoint gradients (`lightsim.invdes.ShapeDesign`) | ✅ (shapes drawn on the design grid; gradient within 0.7 % in 2.5-D and 4 % in 3-D) | ✅ (lumopt, boundary integral) | ❌ |
| Gradients inside the engine: persistent S-matrix session (design sub-block updates, adjoint product on the GPU) and reverse mode through the time loop for time-domain objectives (`lightsim.autodiff`) | ✅ (CPU and GPU) | — | ✅ (autodiff) |
| Nonlinear χ², χ³ (instantaneous, per axis; implicit per-cell Newton solve) | ✅ | ✅ | ❌ |
| Off-diagonal ε and σ tensors (rotated, lossy anisotropic crystals) | ✅ (1 GPU, multi-GPU, multi-node) | ✅ | ✅ |

## In Lumerical FDTD, not in lightsim

The table lists what lightsim has. These Lumerical FDTD capabilities have no
counterpart in lightsim:

- **Gain media and Raman response**: the four-level two-electron laser model,
  current-driven gain and the Raman term of the χ³ model. lightsim's nonlinear
  media are instantaneous χ² and χ³.
- **Magneto-optic media and liquid crystals**: complex (Hermitian)
  permittivity tensors and an orientation that varies from cell to cell.
  lightsim's tensors are real, symmetric and uniform within a medium.
- **Broadband oblique incidence in one run** (BFAST). `lightsim.sweep_angle`
  gives the same spectra from one run per frequency.
- **Zero-thickness sheets.** lightsim represents a sheet by a layer one cell
  thick.
- **User-defined material plugins.**
- **A graphical interface.** lightsim is a Python library, a command line and
  a Rust crate.

Module notes with their validation numbers: [material library](modules/materials.md),
[sheet materials](modules/sheets.md), [shape optimisation](modules/shapes.md),
[index grids](modules/index_grid.md), [gratings](modules/gratings.md),
[resonances](modules/resonance.md) and [effective index](modules/eim.md).
