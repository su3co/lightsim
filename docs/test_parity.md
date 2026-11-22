# FDTDX test-suite parity

This maps every test in FDTDX's simulation suite (`tests/simulation`, FDTDX
main at commit `8d0f69e`, 39 files) to its lightsim counterpart. lightsim
cases live in `crates/lightsim-core/src/validation.rs`. They run on the CPU
(`cargo test -p lightsim-core`) and on GPUs (`lightsim validate --device
cuda:N[,M]`). All 47 cases pass on CPU and on one and two RTX PRO 6000s (fused
and split kernels), and on the GB10 (fused and split kernels).

Legend: ✅ mirrored (same physics, analytic or cross-check reference) · ◐
partly mirrored · ✗ not mirrored (feature gap) · n/a tests FDTDX-specific
machinery that lightsim does not have by design.

## Physics: boundaries

| FDTDX test | lightsim | status |
|---|---|---|
| `test_bloch_oblique` (Fresnel TE at 30°) | `bloch_oblique` (k_x, R_s, T_s; also on 3 CPU slabs and 2 GPUs) | ✅ |
| `test_bloch_zero_vector` (Bloch k=0 ≡ periodic) | `bloch_zero_vector` | ✅ |
| `test_mixed_boundaries::test_pec_pml_waveguide`, `test_pmc_pml_waveguide`, `test_pec_x_pmc_y` | `mixed_boundaries` (TEM wave in a PEC-y / PMC-z channel: k, \|E\|/\|H\|, power) | ✅ |
| `test_mixed_boundaries::test_pec_periodic_channel` | `mixed_boundaries` + `plane_wave_vacuum` (periodic) | ✅ |
| `test_mixed_boundaries::test_bloch_pml_normal` | `bloch_zero_vector` (Bloch k=0 + PML) | ✅ |
| `test_pec_pmc_symmetry` (E node at PEC, antinode at PMC) | `mixed_boundaries` (standing-wave checks) | ✅ |
| `test_pec_reflection`, `test_pmc_reflection` | `pec_pmc_reflection` | ✅ |
| `test_quarter_domain_symmetry` | `symmetry_quarter_domain` (PEC+PMC quadrant vs full domain) | ✅ |
| `test_simulation_symmetry`, `test_symmetry_structured_field` (automatic `config.symmetry` + unfolding, uniform and rectilinear grids) | `lightsim.run_symmetric(sim, (sx, sy, sz))` + Python `test_symmetry`: a dipole with a dielectric sphere on a uniform and a graded mesh. The unfolded DFT fields equal the full run exactly (0 difference) and the box flux matches to 2.5e-5, with 4× fewer cells | ✅ |

## Physics: sources

| FDTDX test | lightsim | status |
|---|---|---|
| `test_dipole_radiation` (power, pattern symmetry, polarization axes, tilted dipole) | `dipole_radiation` (P vs Hertzian formula), `dipole_far_field` (U(θ) ∝ sin²θ, azimuthal symmetry); tilted dipoles are not tested | ◐ |
| `test_dipole_green_function` (P_diel/P_vac = n, near/far field decay) | `dipole_in_dielectric`, `dipole_green_function` | ✅ |
| `test_dipole_magnetic` (radiates, symmetry, duality) | `magnetic_dipole` (P vs analytic) | ✅ |
| `test_gaussian_source` (beam profile, power) | `gaussian_beam` | ✅ |
| `test_mode_source` (n_eff, confinement) | Python `test_python_api` (mode solver vs analytic slab and literature indices, straight-waveguide S21 = 1) | ✅ |
| `test_mode_source` (lossy core → complex n_eff, overlap against the complex permittivity) | `plane_info(freq=...)` gives the complex ε of lossy and dispersive media to the mode solver; Python `test_lossy_mode` (Im n_eff > 0, FDTD decay rate of the injected mode = k₀ Im n_eff within 1.1 %, phase within 0.05 rad over 13 rad) | ✅ |
| `test_plane_source` (amplitude linearity, transverse uniformity) | `plane_wave_vacuum`, `plane_wave_impedance` | ✅ |
| `test_temporal_profiles` | `pulsed_vs_cw` (CW steady state vs pulse vs analytic), Gaussian spectra used by every case | ✅ |
| `test_tfsf_grid_normalization` (dz = 2dx, graded grid) | `anisotropic_cells` (dx:dy:dz = 1:1.5:2, plane wave and TFSF box), `nonuniform_grid` | ✅ |
| `test_tfsf_region` (six-face box, periodic slab, confinement, corners) | `tfsf_leakage` (leakage 1e-7, net flux 1e-14), `mie_scattering`, `anisotropic_cells`; TFSF boxes spanning a periodic axis (slab TFSF) in `test_mie_cylinder` | ✅ |
| `test_tfsf_region::test_dispersive_background_filter_and_run`, `test_dispersive_source` (TFSF in a Lorentz background with broadband impedance correction) | `dispersive_background_sources`: the 1-D incident-field line carries the background's ADE poles, so a TFSF box in a strongly dispersive Lorentz medium leaks 0.8–3.3e-5 across the band (FDTDX filters the H profile instead). Also on 2 CPU slabs and 2 GPUs | ✅ |
| `test_tfsf_region::test_region_gradient_finite` | n/a (lightsim's adjoint does not differentiate through sources) | n/a |

## Physics: materials and monitors

| FDTDX test | lightsim | status |
|---|---|---|
| `test_plane_wave` (phase velocity, impedance in vacuum and ε = 4) | `plane_wave_impedance` | ✅ |
| `test_fresnel` (T, R + T = 1) | `fresnel_normal` | ✅ |
| `test_skin_depth` | `skin_depth` | ✅ |
| `test_dispersion` (Lorentz Fresnel, Drude reflectivity) | `lorentz_slab`, `dispersive_propagation`, `drude_reflection` | ✅ |
| `test_dispersion::test_plane_source_inside_lorentz_medium_has_correct_impedance` | `dispersive_background_sources`: a plane source inside a Lorentz medium radiates only forward (backward power 7e-9 to 3.5e-6 of forward), with impedance \|E\|/η₀\|H\| = 1/\|n(ω)\| within 0.14 % | ✅ |
| `test_anisotropic_dispersion` (per-axis poles, Fresnel per polarization) | `anisotropic_dispersion` | ✅ |
| `test_anisotropic_dispersion::test_hyperbolic_medium_is_polarization_selective` | per-axis Drude is supported but this hyperbolic case is not tested | ◐ |
| `test_birefringence` (k_o, k_e, impedances) | `birefringence` | ✅ |
| `test_oriented_dispersion` (pole along x ≡ per-axis pole, rotated dispersive crystal, monoclinic oscillator pair stable, effective index with off-diagonal coupling) | `oriented_dispersion`: x-oriented pole = per-axis pole (difference 0); dispersive crystal with its oscillator along (x+y)/√2 gives Re n along u and n across u within 0.1 % and the Fresnel ratio within 0.25 %; a monoclinic pair 30° apart decays to 1.5e-3 of its peak; also on 2 CPU slabs and 2 GPUs. `offdiagonal_tensor` covers non-dispersive rotated crystals | ✅ |
| `test_complex_tensor` (rotated lossy uniaxial crystal, full σ tensor) | `lossy_offdiagonal_tensor`: rotated crystal with σ along both crystal axes; complex n_e, n_o of the eigen-polarizations (Re 0.1 %, Im 0.3 %) and the complex Fresnel ratio (0.14 %), along x and z, also on 2 CPU slabs and 2 GPUs | ✅ |
| `test_mie_scattering` (2-D dielectric cylinder) | `two_d_mie_cylinder` (2-D simulation, both polarizations, Q_sca vs the Bohren & Huffman series within 0.1–0.9 %) and `two_d_metal_cylinder`; Python `test_mie_cylinder`: z-invariant cylinder, TFSF box spanning the periodic axis, within 0.1–0.3 %; plus `mie_scattering` (3-D sphere vs BHMIE) | ✅ |
| `test_nonuniform_grid` | `nonuniform_grid` | ✅ |
| `test_phasor_poynting_flux` | `phasor_recording` (phasor flux vs field phasors), `energy_conservation` | ✅ |
| `test_apodization_leakage` | `apodization_leakage` | ✅ |
| `test_mode_overlap` (Gaussian-beam overlap detector: matched is best, waist mismatch follows the two-Gaussian formula, backward reads zero) | `lightsim.gaussian_mode` + Python `test_gaussian_overlap` (η_matched > 0.97, mismatched waists within 0.02 of (2w₁w₂/(w₁²+w₂²))², backward < 1e-3) | ✅ |
| `test_sparams` (waveguide S-parameters) | Python `test_gdsfactory_import_and_straight_sparams`; MMI example (S21 = S31 within 0.002) | ✅ |

## FDTD engine and gradients

| FDTDX test | lightsim | status |
|---|---|---|
| `test_pulsed_vs_cw` | `pulsed_vs_cw` | ✅ |
| `test_detector_region` (sub-sampled DFT ≡ every-step recording) | `phasor_recording` (stride 4 vs stride 1: 1e-7; running DFT vs offline DFT of a probe: 3e-7) | ✅ |
| `test_adjoint_gradient` (adjoint vs finite differences) | Python `test_adjoint` (complex dA/dε within 2 %, 3-D and one-cell-thick grids) | ◐ (one objective type: mode power) |
| `test_time_reversal` (periodic, PEC, PMC, PML with recording, Bloch, lossy strict reversibility, mixed PML + periodic / Bloch with recording) | `tests/time_reversal.rs` (CPU) and `lightsim-cuda/tests/time_reversal_gpu.rs` (GPU, 1–4e-6): `Backend::reverse_e/reverse_h` invert the leapfrog exactly (E^n = (E^{n+1} − cb ∇×H)/ca, H^{n−½} = H^{n+½} + ch ∇×E), and PML layers are replayed from a forward recording. 400 reversed steps reconstruct the fields to 1–3e-6 of the peak (PML only: 4e-9), in all 8 configurations | ✅ |
| `test_fdtd`, `test_reversible_recorder_gradient`, `test_segmented_reversible` (reversible vs checkpointed autodiff gradients agree; finite and non-trivial) | the property they guard, correct gradients, is checked by `test_adjoint` (adjoint vs central finite differences). These tests compare FDTDX's own autodiff memory modes, which lightsim doesn't have: its adjoint is a second forward run. The time reversal those modes rely on is reproduced above | ◐ (gradients checked against finite differences instead) |

## lightsim cases with no FDTDX counterpart

`chi2_second_harmonic`, `chi3_kerr` (FDTDX has no nonlinear media),
`pml_reflection`, `reciprocity`, `energy_conservation`, `stored_energy`
(the energy monitor against the energy of a pulse), `index_grid`,
`dipole_far_field`, `gaussian_beam` (analytic power),
`gaussian_beam_tilted` (direction, power and one-sidedness of a beam at an
angle), domain-decomposition equivalence (`tests/decomposition.rs`,
`lightsim-cuda/tests/multi_gpu.rs`, TCP ranks).

## Summary

The tables above have 42 rows:

- **37** are mirrored with analytic or cross-check references (✅);
- **4** are partly mirrored (◐): tilted dipoles and the hyperbolic medium are
  not tested, the adjoint check covers one type of objective, and the
  reversible-gradient tests are replaced by a check of the gradient against
  finite differences;
- **1** is not applicable (the gradient through a TFSF region).

Time reversal is among the mirrored rows: 8 configurations, on CPU and GPU.
