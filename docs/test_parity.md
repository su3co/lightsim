# FDTDX test-suite parity

This maps every test in FDTDX's simulation suite (`tests/simulation`, FDTDX
main at commit `8d0f69e`, 39 files) to its lightsim counterpart. lightsim
cases live in `crates/lightsim-core/src/validation.rs`. They run on the CPU
(`cargo test -p lightsim-core`) and on GPUs (`lightsim validate --device
cuda:N[,M]`). All 70 cases pass on CPU and on one and two RTX PRO 6000s (fused
and split kernels), and on the GB10 (fused and split kernels); the
double-precision CPU build passes them with the same tolerances.

Legend: ✅ mirrored (same physics, analytic or cross-check reference) · n/a
tests FDTDX-specific machinery that lightsim does not have by design.

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
| `test_dipole_radiation` (power, pattern symmetry, polarization axes, tilted dipole) | `dipole_radiation` (P vs Hertzian formula), `dipole_far_field` (U(θ) ∝ sin²θ, azimuthal symmetry), `dipole_tilted`: x-, y- and z-polarised dipoles and a dipole tilted 45° in the xz plane; the power through each face pair of a cube is within 0.4 % of the share of a sin²θ pattern, the least-flux pair is the dipole's axis, P_x = P_y = P_z to 2e-7, and the tilted dipole radiates the power of the z dipole to 4e-6 | ✅ |
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
| `test_anisotropic_dispersion::test_hyperbolic_medium_is_polarization_selective` | `hyperbolic_drude`: a slab with a Drude pole on x only (Re ε_x < 0 < ε_y); x polarisation reflects like a metal (R within 5e-4 and T within 0.5 % of the slab formula with the discrete ε_x(ω)), y polarisation passes as through a plain dielectric (T within 2e-4) | ✅ |
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
| `test_adjoint_gradient`: dipole source and the power of field phasors, CW and pulsed (`test_dipole_float32`, `test_exact_dipole`); many frequencies in one run (`test_exact_dense_frequencies_in_one_adjoint_run`, `test_frequencies_sharing_one_adjoint_run`); gated, subsampled and apodized windows (`test_exact_detector_windows`) | `reverse_mode_phasor_power_objectives` (CPU), `gpu_reverse_mode_fdtdx_parity_scenes` (GPU), Python `test_reverse_mode_phasor_power`: gradient against central finite differences within 8e-5 on the CPU and 3e-4 on the GPU at 3 and at 12 frequencies, 2e-4 with a gated, strided, Tukey-windowed recording, 1e-5 with CW and pulsed sources together | ✅ |
| `test_adjoint_gradient::test_finite_differences_float64` (gradient against finite differences of the simulated loss) | `reverse_mode_matches_finite_differences` (PEC / PMC / periodic walls, PML, lossy design, Lorentz + Drude, Debye, design next to the CPML: 2e-6 to 2e-4), `reverse_mode_through_a_kerr_medium`, Python `test_adjoint` (two-run adjoint, complex dA/dε within 2 %), `test_native_session_finite_differences`, `test_complex_smatrix_vjp_2x2_multifrequency` | ✅ |
| `test_adjoint_gradient::test_nonuniform_grid` | `reverse_mode_on_a_graded_mesh`: flux objective 4e-5, phasor objective 1e-5 | ✅ |
| `test_adjoint_gradient::test_several_devices_with_overlapping_halos`, `test_gradient_region_without_device`, `test_gradient_regions_device_and_object` | `reverse_mode_with_two_adjacent_grids_and_a_uniform_block`, Python `test_two_adjacent_design_grids_and_a_block`: two design grids one cell apart and a uniform block (a one-sample grid); all together 3e-5, each alone 2e-4; the block through the S-matrix session within 2.2 % | ✅ |
| `test_adjoint_gradient::test_periodic_plane_source_and_flux` | `reverse_mode_periodic_cell_plane_wave_flux`: plane wave in a periodic cell, design spanning the cell; flux at one frequency 9e-6, energy 6e-6 | ✅ |
| `test_adjoint_gradient::test_mode_source_and_mode_overlap` | Python `test_adjoint`, `test_native_session_finite_differences` (mode source, mode-overlap S-parameters: within 0.9 % of the largest derivative); `reverse_mode_and_session_agree_on_one_objective` (a mode-overlap objective through reverse mode: 9e-5) | ✅ |
| `test_adjoint_gradient::test_mirror_symmetry` | `reverse_mode_with_mirror_symmetry`, Python `test_reverse_mode_symmetry_reduced_domain`: the gradient on the half domain equals the full-domain gradient to 1e-7 | ✅ |
| `test_adjoint_gradient::test_anisotropic_and_conductive_background` | `reverse_mode_with_anisotropic_and_conductive_media` (1e-5, conductivities up to 1.5e4 S/m). At FDTDX's 2–3e4 S/m the S-matrix session agrees with finite differences to 0.5 %; reverse mode refuses such a run with an error (`reverse_mode_refuses_a_diverged_reverse_pass`), because a strong absorber cannot be stepped backwards | ✅ |
| `test_adjoint_gradient::test_gradient_region_overlapping_a_source_raises`, `test_device_in_pml_raises` | `reverse_mode_refuses_a_source_on_the_design`, `design_grids_must_stay_out_of_the_pml`: both set-ups are refused with an error | ✅ |
| `test_adjoint_gradient`: the double-precision variants (`test_dipole_float64`, `test_exact_dipole[float64]`, `test_finite_differences_float64`) | in the double-precision CPU build (`--features f64`): `f64_reverse_mode_matches_finite_differences` (seven scenes: walls, CPML, lossy design, Lorentz + Drude, Debye, χ² + χ³, design up to the PML) within 2e-10 of extrapolated central differences, `f64_reverse_smatrix_matches_finite_differences` within 2e-11, Python `test_reverse_engine_double_precision` within 3e-11; all three assert 1e-8 | ✅ |
| `test_adjoint_gradient::test_sharded_over_two_devices` (the fields of one simulation split over two devices) | Python `test_python_engine_runs_split_over_two_gpus`: with `engine="python"` and `devices="cuda:0,1"` every forward and adjoint run is split over two RTX PRO 6000; S and the gradient equal the one-GPU result bit for bit. The native session and reverse mode run each simulation on one device | ✅ |
| `test_adjoint_gradient::test_inverse_permeability_gradient_inside_devices`; `test_fdtd::TestReversibleVsCheckpointedGradient` with magnetic, lossy media | `reverse_mode_permeability_gradient` (CPU), `gpu_reverse_mode_permeability_gradient`, Python `test_reverse_mode_permeability_gradient`: a design grid that carries the permeability, behind a μ = 1.5 slab and behind a slab with magnetic conductivity; gradient against central finite differences within 4e-4. The magnetic conductivity itself is checked by the `magnetic_loss` case. Reverse mode only: the S-matrix session refuses permeability grids | ✅ |
| `test_adjoint_gradient::test_final_field_in_loss_raises`, `test_mixed_cw_and_pulsed_sources_raise`, `test_nonlinear_detector_in_loss_raises`, `test_unused_nonlinear_detector_is_fine` | n/a: limits of FDTDX's frequency-domain adjoint. lightsim's reverse mode differentiates these objectives: the final fields in a periodic box (1.5e-5), CW and pulsed sources together (1.2e-5), time-domain energies | n/a |
| `test_adjoint_gradient`: option checks (`test_options_of_the_mode_that_does_not_run_raise`, `test_adjoint_frequencies_override`, `test_without_device_a_gradient_region_is_required`), `test_progress_bar_of_the_adjoint_run`, `test_arrays_as_jit_arguments` | n/a: FDTDX's options, its progress bar and JAX argument passing | n/a |
| `test_time_reversal` (periodic, PEC, PMC, PML with recording, Bloch, lossy strict reversibility, mixed PML + periodic / Bloch with recording) | `tests/time_reversal.rs` (CPU) and `lightsim-cuda/tests/time_reversal_gpu.rs` (GPU, 1–4e-6): `Backend::reverse_e/reverse_h` invert the leapfrog exactly (E^n = (E^{n+1} − cb ∇×H)/ca, H^{n−½} = H^{n+½} + ch ∇×E), and PML layers are replayed from a forward recording. 400 reversed steps reconstruct the fields to 1–3e-6 of the peak (PML only: 4e-9), in all 8 configurations | ✅ |
| `test_fdtd::TestReversibleFdtdGradients` (finite, non-trivial), `TestReversibleVsCheckpointedGradient` (two gradient routes agree) | every gradient check asserts a finite, non-zero gradient. `reverse_mode_and_session_agree_on_one_objective`: reverse mode through the time loop and the two-run adjoint of the S-matrix session, two independent routes, agree within 0.14 % along a random direction and 0.8 % of the largest entry (reverse mode against finite differences: 9e-5; the rest is the discretisation error of the continuous adjoint) | ✅ |
| `test_fdtd::TestPerAxisDispersiveGradientCheckpointed` | `reverse_mode_pole_coefficient_gradients` (CPU), `gpu_reverse_mode_pole_coefficient_gradients`, Python `test_reverse_mode_pole_gradients`: gradient with respect to the parameters of per-axis Lorentz and Debye poles (Δε, ω₀, γ, τ) against extrapolated finite differences within 5e-4; `reverse_mode_next_to_per_axis_dispersion`: the permittivity gradient through such a medium (1e-6) | ✅ |
| `test_fdtd::TestOrientedDispersionGradients` (the gradient is finite and non-trivial with an oriented dispersive medium) | `session_through_an_oriented_dispersive_medium` (CPU), `gpu_session_through_an_oriented_dispersive_medium`, Python `test_session_through_an_oriented_dispersive_medium`: the S-matrix gradient with an oriented Lorentz slab next to the design is finite, non-zero and within 0.35 % of central finite differences. Reverse mode refuses oriented poles (`reverse_mode_refuses_oriented_dispersion`) | ✅ |
| `test_fdtd::TestSlicedReversibleGradient`, `test_segmented_reversible`, `test_reversible_recorder_gradient` | n/a: FDTDX's memory modes (checkpoint slices, a compressing recorder). What they protect, a correct gradient when the absorbing layers are replayed from a recording, is checked by the PML cases above (2e-5) and by `reverse_pass_reconstructs_the_forward_fields` (3e-7) | n/a |

## lightsim cases with no FDTDX counterpart

`chi2_second_harmonic`, `chi3_kerr` (FDTDX has no nonlinear media),
`pml_reflection`, `reciprocity`, `energy_conservation`, `stored_energy`
(the energy monitor against the energy of a pulse), `index_grid`,
`dipole_far_field`, `gaussian_beam` (analytic power),
`gaussian_beam_tilted` (direction, power and one-sidedness of a beam at an
angle), `bloch_oblique_p` (p-polarised oblique plane wave in 3-D: Fresnel at
30° and Brewster's angle), domain-decomposition equivalence (`tests/decomposition.rs`,
`lightsim-cuda/tests/multi_gpu.rs`, TCP ranks).

## Summary

The tables above have 58 rows:

- **54** are mirrored with analytic or cross-check references (✅): every
  physics test (boundaries, sources, materials and monitors) and every
  gradient test that checks a property of the gradient, the double-precision
  ones in the double-precision CPU build;
- **4** are not applicable: the gradient through a TFSF region, and tests of
  FDTDX's own machinery (its adjoint's limits, options, memory modes).

Some mirrored rows hold for one of lightsim's gradient routes and not the
other, as each row says: permeability and pole-parameter gradients come from
reverse mode, gradients through oriented dispersive media and runs split
over two GPUs from the S-matrix engines.

Time reversal is among the mirrored rows: 8 configurations, on CPU and GPU.
