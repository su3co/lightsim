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
| magnetic_loss | magnetic conductivity: a half-space with matched electric and magnetic loss (σ*/μ₀μ = σ/ε₀ε) reflects nothing and decays as exp(−ση₀x) with \|E\|/η₀\|H\| = 1; a half-space with magnetic loss only: R = \|(Z−1)/(Z+1)\|², decay k₀ Im Z, impedance \|Z\| | R 1e-3 (5 % of the 2e-2 the same σ alone reflects); decay 0.3–0.4 %; impedance 1e-6–7e-5; R 1–2 % |
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
| dipole_tilted | x-, y-, z-polarised dipoles and a dipole tilted 45° in the xz plane (Ex + Ez with the same waveform): power through each face pair of a cube vs the analytic sin²θ share (0.1496 axial, 0.4252 transverse; tilted 0.2874 x and z, 0.4252 y), least flux on the axial pair, P vs η₀k²\|Il\|²/12π, P_x = P_y = P_z, tilted P vs P_z with the mutual term of the half-cell Ex–Ez offset, x faces = z faces | shares 0.05–0.4 %, P 0.4–1 %, P_x/P_z 2e-7, tilted/P_z 4e-6, x/z 2e-7 |
| bloch_oblique | k_x = k₀cosθ, oblique s-pol Fresnel R, T (Bloch k_y = k₀ sinθ) | 0.04 %, R 1.5 %, T 0.1 % |
| bloch_oblique_p | p-polarised (H along z) plane wave, Bloch k_y = k₀ sinθ: k_x = k₀cosθ, E_y = A cosθ and E_x = A sinθ of the incident wave, Fresnel R_p, T_p of an n = 2 half-space at 30° and at Brewster's angle | k 0.04 %, E 0.01 %, R 3 %, T 0.3 %, R_p at Brewster 8e-6 |
| chi2_second_harmonic | χ² slab: SH spectrum vs forward-wave theory −(χL/2c)∂ₜE² | 0.6–0.9 % |
| chi3_kerr | χ³ slab: self-phase modulation and third harmonic vs −(χL/2c)∂ₜE³ | SPM 0.07 %, THG 0.9 % |
| chi3_raman | χ³ with a Raman response: field radiated by a thin slab under a pulse vs −(L/2c)∂ₜP with P = χ³E[(1−f)E² + f g_R∗E²] (analytic g_R(t), f = 1 and 0.5, 3-D and 2-D); stimulated Raman gain of a Stokes probe under a CW pump vs the coupled-wave coefficient g = (ω_s/c)(fχ³/2)A_p² Im g_R(Ω) | 0.1–0.2 % (THG 0.9 %); gain 0.9 % at gL = 0.02 |
| offdiagonal_tensor | uniaxial crystal rotated 45° about the propagation axis (x and z): n_e, n_o of the rotated eigen-polarizations, per-mode Fresnel ratio | n 0.06–0.09 %, ratio 0.06 % |
| lossy_offdiagonal_tensor | the same crystal with σ along both axes: complex n_e, n_o and complex Fresnel ratio | Re 0.1 %, Im 0.3 %, ratio 0.14 % |
| oriented_dispersion | pole along x ≡ per-axis pole; dispersive crystal with oscillator along (x+y)/√2: n along/across u, Fresnel ratio; monoclinic pair stable | exact, 0.1 %, 0.25 %, decays to 1.5e-3 |
| director_field | director field (liquid crystal): uniform director vs the rotated tensor of offdiagonal_tensor (x and z propagation); 90° twisted nematic layer (n_o 1.5, n_e 1.6, 10 µm) between index-matched half-spaces: transmitted Jones vector vs the Jones-matrix product of 4000 rotated retarders at 0.8–1.2 f₀ (magnitudes, relative phase, ellipse azimuth), 3-D and 2-D | exact (0); magnitudes 2e-3, phase 3–11 mrad, azimuth 0.1° |
| gain_four_level | four-level gain medium: transmission of an inverted slab vs the slab formula with the frozen populations' discrete χ(ω) (3-D and 2-D); steady-state gain of a pumped slab at 0.01, 1 and 3 times the saturation intensity vs the rate equations integrated through the slab; field energy taken by an absorbing slab (Poynting flux) vs ħω times the atoms in level 2; atom number | T 0.05–0.2 %, saturation 0.1–0.17 %, energy 5e-7, atoms 1e-13 |
| gyrotropic_faraday | magneto-optic (gyrotropic) poles, bias along the propagation axis: indices n± of the circular polarisations and Faraday rotation per length in a half-space vs the scheme's discrete ε(ω) tensor, at 0.9, 1.0, 1.1 f₀ (Lorentz pole biased along z, Drude pole biased along x, 2-D); rotation crossing backwards vs forwards (same sense in the lab frame: non-reciprocal); a magnetised plasma stays bounded over 600 fs | n 0.02–0.1 %, rotation 0.1–0.5 %, backwards/forwards 3e-4, late field 5e-7 |
| dispersive_background_sources | TFSF box and plane-wave source inside a strongly dispersive Lorentz medium: scattered-field leakage, backward power, impedance 1/\|n(ω)\| | 3e-5, 4e-6, 0.14 % |
| plane_wave_impedance | k and \|E\|/η₀\|H\| in vacuum and n = 2 | k 0.17 %, Z 6e-5 |
| mixed_boundaries | TEM wave in a PEC-y / PMC-z channel (k, Z, power); E node at PEC / antinode at PMC | 0.1–0.7 % |
| dipole_in_dielectric | P_diel = n · P_vac | 0.5–1.1 % |
| drude_reflection | Drude half-space (f_p = 4 f₀) R vs Fresnel with discrete ε(ω) | 2–5e-4 |
| hyperbolic_drude | hyperbolic slab (Drude pole on x only, f_p = 4 f₀, ε_∞ = 2.25: Re ε_x < 0 < ε_y), 60 nm, normal incidence: R, T of x polarization vs slab Fresnel with discrete ε_x(ω), of y polarization vs ε_∞; T_y − T_x > 0.8 | x: R 3–5e-4, T 0.4–0.5 %; y: R 0.06–0.1 %, T 1–2e-4 |
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
| fixed_angle_slab | broadband oblique incidence at a fixed angle (`fixed_angle`), 2-D: R, T of an n = 1.5 slab at 13 frequencies over ±30 % from one run per case, at 30° and 60°, s and p, against the Airy formula at that angle; narrow-band Bloch runs at 0.7, 1, 1.3 f₀ for comparison; R + T; incident power vs \|S\|² cosθ/2η₀; backward leakage of the source | R, T 3.6e-3 / 2.2e-3 (30° s / p), 9.3e-3 / 7.1e-4 (60°); Bloch runs 3.6e-3 (30°), 7.3e-3 (60°); R + T 1.9e-4; power 1.3–3.9e-3; leakage 1e-9 |
| fixed_angle_brewster | p polarisation at Brewster's angle of an n = 2 half-space, one fixed-angle run over ±30 %: R at every frequency | ≤ 6.3e-5 |
| fixed_angle_dispersive | Lorentz slab (60 cells per wavelength) and Drude film (f_p = 4 f₀, 30 nm, dx = 5 nm) at 45°, s and p, over ±30 %: R, T vs Airy with the permittivity of the trapezoidal pole update | Lorentz ≤ 7.2e-3, Drude ≤ 1.0e-3 |
| fixed_angle_grating | lamellar grating at 15°, both polarisations: efficiency of every propagating order at 0.8, 0.95, 1.1, 1.2 f₀ from one fixed-angle run vs narrow-band Bloch runs at each frequency (orders appear at 0.90 and 1.06 f₀; orders up to 79° from the normal) | ≤ 2.1e-4 per order; orders sum to 1 within 3e-5 |
| fixed_angle_stability | 80°, s and p, dielectric, Drude and Lorentz layers, run ten times longer than the pulse (5e4 steps, no shutoff) | energy at the end 5e-10 of its peak, ≤ 2.5e-8 over the second half |
| fixed_angle_3d | 3-D, 45°, s and p: tilt along x on an n = 1.5 slab (R, T vs Airy; transmitted field at f₀ vs a Bloch run), tilt along y (the other handedness) on a Lorentz slab (R, T vs Airy at 30 cells per wavelength; energy left in the empty run) | R, T 1.4e-2 (s), 3.5e-3 (p); field 1.3e-4, phase 9e-4 rad; Lorentz 3.2e-2; 7e-13 |
| sheet_normal_incidence | zero-thickness sheets (a Drude sheet, and one with conductance, susceptibility, Drude and Lorentz terms) between n = 1.4 and n = 2 at normal incidence: R, T vs the sheet formulas at λ/20, λ/40, λ/80 with the sheet off the nodes of the uniform mesh (the mesh puts a node on it), on explicit uniform meshes (sheet shared by the two planes around it) and on an explicit mesh whose cells change size at the sheet; R + T + A = 1 with A from the sheet current | aligned: 5–6e-3, 0.9–1.8e-3, 1.7–3.4e-4 (second order); between nodes: 1.4e-2, 6.9e-3, 2.2e-3 (first order); non-uniform 1.5e-3; energy 2e-5–9e-4 |
| sheet_oblique | 2-D, Bloch k_y: R, T of a sheet with every term between vacuum and n = 1.5, s and p at 30° and 60°, vs the oblique sheet formulas | R 1e-4–4e-4, T 2.3–3.5e-3 |
| sheet_plasmon | 2-D sheet plasmon of a Drude sheet between vacuum and ε = 2.25 (q ≈ 8 k₀): complex wavenumber fitted from the field on the sheet vs the retarded dispersion relation, 0.85, 1.0, 1.15 f₀, at 30 and 60 cells per plasmon wavelength | Re q 0.4–1.3 % and 0.005–0.26 % (second order), Im q 0.4–1.5 % |
| sheet_ribbon | periodic array of sheet strips (2-D patches, w = 20 nm, period 60 nm, on ε = 2.25), E across the strips: absorption resonance vs the quasi-static eigenvalue of the strip array, 20 and 40 cells across the strip | 2.1 %, 1.8 % (0.6 % at 80 cells, first order) |
| plugin_media | user-defined material plug-ins: one Lorentz pole (conductive background, and lossless with the slab faces between nodes) and the instantaneous Kerr medium written as plug-ins vs the built-in pole and χ³ medium (field behind and inside a slab); a plug-in that amplifies the field each step is stopped by the divergence check | identical on the CPU (0); stopped with an error |
| gain_two_electron | four-level two-electron gain (Chang and Taflove): populations of a plane of atoms under a CW pump vs the continuous model integrated by RK4 from the recorded field; probe gain at ω_a and pump attenuation of a thin slab in its steady state vs the steady-state rate equations with the discrete Lorentzians; 1-D slab laser at 1.4, 1.7, 2 × threshold: steady single line at the frequency of the threshold condition, pump threshold from the output powers vs the rate equations, inversion nearly clamped; electron number | populations 2e-5 N_d; gain 0.1 %, pump 0.2 %; lasing frequency 0.14–0.17 %, steady to 0.6 %, threshold 6 %, inversion +7.5 % from 1.4 to 2 × threshold; electrons 1e-13 |

Plus 35 unit tests (grid, rasterizer incl. polygon unions, materials, Mie, …)
and the Python suite (`pytest python/tests`: mode solver vs analytic slab and
literature strip-waveguide indices, gdsfactory import, straight-waveguide
S-parameters, plane-wave flux through the API, adjoint gradient vs finite
differences — complex dA/dε within 2 % in magnitude and 0.003 rad in phase,
on 3-D grids and on grids one cell thick;
Gaussian-beam overlap vs the two-Gaussian formula; lossy-core waveguide with
complex n_eff whose FDTD decay matches k₀ Im n_eff within 1.1 %; automatic
symmetry reduction reproducing the full domain exactly; 2-D Mie scattering by
a cylinder within 0.1–0.3 % of the Bohren & Huffman series; conducting sheets
and graphene within 2e-3 of the analytic sheet transmission; spectra of a slab
at a fixed 45° within 5e-3 of the Airy formula, s- and p-polarised; shape
gradients against finite differences within 0.1 % with the exact reverse
engine, and within 0.7 % in 2.5-D and 4 % in 3-D with the adjoint engines).
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

All 55 physics cases, eight of them two-dimensional, pass on CPU, on the RTX
PRO 6000 with both the split and the fused GPU kernels (also with the dense
dispersive path forced, `LIGHTSIM_DISP_DENSE=1`), split across both RTX PRO
6000s (`--device cuda:0,1`, fused and split) and on the GB10 (fused and split). The Python
suite (237 tests, one of which needs two GPUs and one the double-precision build, with the inverse-design and autodiff checks of
[modules/invdes.md](modules/invdes.md) and [modules/autodiff.md](modules/autodiff.md))
passes on CPU and on the RTX PRO 6000. A double-precision CPU build (`--features f64`) passes the same 55 cases with unchanged
tolerances. Every physics test of FDTDX's simulation suite has a lightsim
counterpart ([test_parity.md](test_parity.md)).

The six fixed-angle cases (`fixed_angle_*`, 61 cases in all) pass on CPU,
on the RTX PRO 6000 (fused and split kernels: a fixed-angle run always uses
the split kernels), on both RTX PRO 6000s (`--device cuda:0,1`: the 2-D cases,
which propagate along x, run split over the two GPUs; `fixed_angle_3d`
propagates along z and runs on the first GPU), on the GB10 (fused and split)
and in the double-precision CPU build. `tests/fixed_angle.rs` checks the
time-step rule, the refusals (also by the three gradient engines) and that
three CPU x-slabs reproduce one bit for bit; `lightsim-cuda/tests/fixed_angle_gpu.rs`
checks one GPU against the CPU (5e-6) and two GPUs against one.
