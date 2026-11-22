# Validation

`cargo test -p lightsim-core` runs every case on the CPU reference engine and
`lightsim validate --device cuda:N` on a GPU. The cases mirror FDTDX's physics
test-suite (`tests/simulation/physics`) and classic analytic checks; the
test-by-test map, including the parts not mirrored yet, is in
[test_parity.md](test_parity.md):

| case | reference | typical error |
|---|---|---|
| plane_wave_vacuum | analytic incident flux; scattered-field side dark | 0.4–0.7 % (flux colocation, O(dx²)); leakage 1e-8 |
| fresnel_normal | R, T of an n = 2 interface | T 1e-5, R 1 % |
| pec_pmc_reflection | \|R\| = 1 | 0.5 % |
| dipole_radiation / magnetic_dipole | P = η₀k²\|Il\|²/12π, k²\|K\|²/12πη₀ | 0.4–1 % |
| dipole_green_function | near/far field E_z(r) ratio and phase | 0.25 %, 4 mrad |
| pml_reflection | small vs large domain | 2.4e-5 |
| skin_depth | conductor decay length | 0.25 % |
| lorentz_slab | transfer matrix with discrete χ(ω) | ≤ 0.005 abs |
| dispersive_propagation / anisotropic_dispersion | Re n(ω) | ≤ 0.2 % |
| birefringence | n_y, n_z of a uniaxial crystal | 0.1 % |
| nonuniform_grid | graded vs uniform mesh | 6e-5 |
| symmetry_quarter_domain | PMC-symmetric quadrant vs full domain | exact |
| tfsf_leakage | box confinement, incident amplitude | leakage 1e-7, net flux 1e-14 |
| mie_scattering | Mie theory (BHMIE) σ_sca | 0.5–0.9 % |
| gaussian_beam | analytic beam power | 1–1.6 % |
| gaussian_beam_tilted | beam at 20° polar angle and 30° azimuth, focus beyond the plane: direction from the intensity centroid on two planes, analytic power, power behind the source | 0.7°, 1.9 %, 9e-4 |
| energy_conservation | zero net flux around a lossless scatterer | 1e-6 of face power |
| stored_energy | energy monitor: a pulse inside a box holds ε₀ n A c ∫s² dt, in vacuum and in n = 2; nothing is left after it has gone | 0.4–0.5 %, 1e-10 |
| reciprocity | T forward = T backward | 4e-7 |
| bloch_zero_vector | Bloch(k = 0) ≡ periodic | exact |
| pulsed_vs_cw | CW steady state vs pulsed vs analytic | 0.02–0.16 % |
| apodization_leakage | windowed DFT suppresses leakage | 7.5× |
| dipole_far_field | U(θ) ∝ sin²θ, ∮U dΩ = box flux | 0.2 %, 2e-4 |
| bloch_oblique | k_x = k₀cosθ, oblique s-pol Fresnel R, T (Bloch k_y = k₀ sinθ) | 0.04 %, R 1.5 %, T 0.1 % |
| chi2_second_harmonic | χ² slab: SH spectrum vs forward-wave theory −(χL/2c)∂ₜE² | 0.6–0.9 % |
| chi3_kerr | χ³ slab: self-phase modulation and third harmonic vs −(χL/2c)∂ₜE³ | SPM 0.07 %, THG 0.9 % |
| offdiagonal_tensor | uniaxial crystal rotated 45° about the propagation axis (x and z): n_e, n_o of the rotated eigen-polarizations, per-mode Fresnel ratio | n 0.06–0.09 %, ratio 0.06 % |
| lossy_offdiagonal_tensor | the same crystal with σ along both axes: complex n_e, n_o and complex Fresnel ratio | Re 0.1 %, Im 0.3 %, ratio 0.14 % |
| oriented_dispersion | pole along x ≡ per-axis pole; dispersive crystal with oscillator along (x+y)/√2: n along/across u, Fresnel ratio; monoclinic pair stable | exact, 0.1 %, 0.25 %, decays to 1.5e-3 |
| dispersive_background_sources | TFSF box and plane-wave source inside a strongly dispersive Lorentz medium: scattered-field leakage, backward power, impedance 1/\|n(ω)\| | 3e-5, 4e-6, 0.14 % |
| plane_wave_impedance | k and \|E\|/η₀\|H\| in vacuum and n = 2 | k 0.17 %, Z 6e-5 |
| mixed_boundaries | TEM wave in a PEC-y / PMC-z channel (k, Z, power); E node at PEC / antinode at PMC | 0.1–0.7 % |
| dipole_in_dielectric | P_diel = n · P_vac | 0.5–1.1 % |
| drude_reflection | Drude half-space (f_p = 4 f₀) R vs Fresnel with discrete ε(ω) | 2–5e-4 |
| anisotropic_cells | plane-wave power and TFSF amplitude with dx:dy:dz = 1:1.5:2 | 0.4–0.7 %, TFSF 7e-7 |
| phasor_recording | stride-4 DFT vs every step, running vs offline DFT, phasor flux vs field phasors | 2e-7, 3e-7, 0.4–0.7 % |
| two_d_matches_one_cell | 54 small 2-D scenes (three invariant axes, both field sets, CPML, PEC / PMC, periodic and Bloch boundaries, lossy, dispersive and nonlinear media, dipoles, plane waves at normal and oblique incidence, TFSF): the 2-D update against the general 3-D update on the same one-cell grid | exact on the CPU, 3e-7 of the peak field on the GPU |
| two_d_fresnel_oblique | 2-D, Bloch k_y: Fresnel R and T at 30° and at Brewster's angle, s (TM set) and p (TE set) polarization | R 1–2.4 %, T 0.02–0.2 %, R_p at Brewster 3e-6 |
| two_d_mie_cylinder | 2-D Mie series (Bohren & Huffman): Q_sca of a dielectric cylinder (n = 2), E along the axis and across it, five frequencies | 0.1–0.9 % |
| two_d_metal_cylinder | the same for a perfectly conducting cylinder (staircased; first-order convergence checked at 40 to 320 cells per wavelength) | 0.8 % (E along the axis), 2.2 % (E across) at 160 cells per wavelength |
| two_d_slab_mode | n_eff of the fundamental mode of a slab waveguide from the phase along the core, against the dispersion relation, slab TE and TM | 0.07 %, 0.12 % |
| two_d_cpml | line source next to the CPML, small vs large domain, both sets | 1.4e-5 |
| two_d_grating | order efficiencies of a lamellar grating at 15° (three reflected and four transmitted orders), both sets, against RCWA | ≤ 0.6 % of the incident power per order; orders sum to 1 within 1e-5 |
| two_d_gaussian_beam | 2-D (sheet) Gaussian beam, both sets: power with the waist on the source plane, 0.5/η₀ \|A S\|² w₀ √(π/2) per unit thickness; radius and power at the focus of a beam focused 4 µm away | 0.9–1.6 %; radius 1 % |

Plus 30 unit tests (grid, rasterizer incl. polygon unions, materials, Mie, …)
and the Python suite (`pytest python/tests`: mode solver vs analytic slab and
literature strip-waveguide indices, gdsfactory import, straight-waveguide
S-parameters, plane-wave flux through the API, adjoint gradient vs finite
differences — complex dA/dε within 2 % in magnitude and 0.003 rad in phase,
on 3-D grids and on grids one cell thick;
Gaussian-beam overlap vs the two-Gaussian formula; lossy-core waveguide with
complex n_eff whose FDTD decay matches k₀ Im n_eff within 1.1 %; automatic
symmetry reduction reproducing the full domain exactly; 2-D Mie scattering by
a cylinder within 0.1–0.3 % of the Bohren & Huffman series).
Time-reversal tests (`tests/time_reversal.rs` on CPU,
`lightsim-cuda/tests/time_reversal_gpu.rs` on GPU, after FDTDX's) step 400
times backwards and reconstruct the fields to 1–4e-6 of the peak with periodic,
PEC, PMC, Bloch, lossy, PML (recorded) and mixed boundaries.
Domain-decomposition tests check that x-slab runs reproduce single-slab runs:
CPU slabs (`tests/decomposition.rs`: Bloch k ≠ 0 along x and y incl. the phased
x wrap, oblique Bloch incidence, χ² on three slabs, rotated crystals on two
slabs through the dD exchange), two GPUs
(`lightsim-cuda/tests/multi_gpu.rs`: 1 vs 2 GPUs within 5e-7 of the peak field)
and three TCP ranks (bit-identical on CPU). On three GB10 nodes over direct
200 GbE links, a Bloch (k ≠ 0, phased x wrap) box matches a single GB10 within
5e-7 of the peak field and a rotated-crystal column (dD exchange) matches exactly.

All 47 physics cases, eight of them two-dimensional, pass on CPU, on the RTX
PRO 6000 with both the split and the fused GPU kernels (also with the dense
dispersive path forced, `LIGHTSIM_DISP_DENSE=1`), split across both RTX PRO
6000s (`--device cuda:0,1`, fused and split) and on the GB10 (fused and split). The Python
suite (158 tests, with the inverse-design and autodiff checks of
[modules/invdes.md](modules/invdes.md) and [modules/autodiff.md](modules/autodiff.md))
passes on CPU and on the RTX PRO 6000. Every physics test of FDTDX's simulation suite has a lightsim
counterpart ([test_parity.md](test_parity.md)).
