# Port profiles over a band

A mode profile solved at the centre frequency is used at every frequency of a
port or mode monitor (`port_profiles="center"`, the default). With
`port_profiles="band"` (`lightsim.profile_over_band`) the mode is solved at
two or three more frequencies across the band and a monitor projects, at each
frequency, on the profile interpolated between them. This folder holds the
measurements behind that option. All runs: silicon (ε = 12.25) in oxide
(1.444²), 1.50 to 1.60 µm, profile of 1.55 µm, CPU engine in single
precision, Apple M1 Pro.

"Three frequencies" below is the option; "every frequency" solves the mode at
each monitor frequency (`nodes="all"`). The two agree to every digit printed,
so the option costs two or three mode solves per port whatever the number of
frequencies.

## The share of a wave read as travelling the other way

A guide that ends on a conducting wall returns its mode whole. On a row of
planes in front of the wall the forward amplitude should have one modulus;
its ripple is the share of the returning wave that the monitor reads as
forward (`guide_on_a_wall_2d.py`, `wire_on_a_wall_3d.py`).

| device, cell | profile | 1.50 µm | 1.525 µm | 1.55 µm | 1.575 µm | 1.60 µm |
|---|---|---|---|---|---|---|
| 2-D guide, E in plane, 45 nm | centre | 0.390 % | 0.228 % | 0.053 % | 0.123 % | 0.300 % |
| | three frequencies | 0.062 % | 0.059 % | 0.053 % | 0.048 % | 0.045 % |
| 2-D guide, E in plane, 20 nm | centre | 0.320 % | 0.161 % | 0.001 % | 0.162 % | 0.331 % |
| | three frequencies | 0.002 % | 0.001 % | 0.001 % | 0.001 % | 0.001 % |
| 2-D guide, E out of plane, 45 nm | centre | 0.056 % | 0.010 % | 0.072 % | 0.137 % | 0.201 % |
| | three frequencies | 0.080 % | 0.077 % | 0.072 % | 0.068 % | 0.063 % |
| 480 × 220 nm wire, 40 nm | centre | 0.157 % | 0.086 % | 0.011 % | 0.093 % | 0.200 % |
| | three frequencies | 0.010 % | 0.007 % | 0.011 % | 0.020 % | 0.028 % |
| 480 × 220 nm wire, 20 nm | centre | 0.125 % | 0.067 % | 0.0004 % | 0.075 % | 0.159 % |
| | three frequencies | 0.0015 % | 0.0008 % | 0.0004 % | 0.0018 % | 0.0021 % |

What remains with the band profile is the floor of the monitor at that cell
size (it falls as the square of the cell or faster).

## Energy balance of a reflective device

`closed_guide_balance.py`: a 225 nm guide between two conducting plates
1.08 µm apart (E out of plane), cut by a 180 nm gap. Nothing can radiate and
one mode of the gap's symmetry propagates, so |S11|² + |S21|² = 1;
|S11| is 0.61 to 0.63. Largest deviation over nine wavelengths:

| cell | centre | band |
|---|---|---|
| 45 nm | 0.72 % | 0.157 % |
| 22.5 nm | 0.66 % | 0.018 % |

In 3-D the wire on the wall gives |S11|² − 1 on the worst plane: 0.151 %
(centre) and 0.088 % (band) at 40 nm, 0.064 % and 0.017 % at 20 nm.

## The two normalisations of the inverse-design problem

`two_normalisations_2d.py`: the 2-D inverse-design benchmark scene with eight
fixed designs (|S11| 0.2 to 0.57) at 1.50, 1.55 and 1.60 µm. A fixed source
launches the same amplitude whatever the design, so S with
`normalize="monitor"` over S with `normalize="source"` must be one constant
per wavelength (taken from a straight guide). Spread of that ratio over the
designs, and the reflection reported for the straight guide:

| cell | profiles | spread (std) | range | straight guide, \|S11\| at 1.50 / 1.55 / 1.60 µm |
|---|---|---|---|---|
| 45 nm | centre | 0.19 % | −0.42 to +0.36 % | 0.0032 / 0.0005 / 0.0042 |
| 45 nm | band | 0.04 % | −0.07 to +0.08 % | 0.0006 / 0.0005 / 0.0004 |
| 22.5 nm | centre | 0.14 % | −0.38 to +0.19 % | 0.0039 / 0.00004 / 0.0040 |
| 22.5 nm | band | 0.00 % | −0.00 to +0.00 % | 0.00005 / 0.00004 / 0.00004 |

The Python and the native engine give the same numbers.

## Gradients

`gradient_check_2d.py`: gradient of |S21|² + |S11|²/2 of a grey design on the
same scene against central finite differences, per wavelength: along a smooth
direction over the whole design / worst of four single samples (relative to
the largest derivative).

| engine | profiles | 1.50 µm | 1.55 µm | 1.60 µm |
|---|---|---|---|---|
| reverse | centre | 0.000 % / 0.003 % | 0.000 % / 0.004 % | 0.003 % / 0.001 % |
| reverse | band | 0.000 % / 0.003 % | 0.000 % / 0.004 % | 0.003 % / 0.001 % |
| python, native | centre | 0.26 % / 0.22 % | 0.21 % / 0.09 % | 0.68 % / 0.07 % |
| python, native | band | 0.29 % / 0.36 % | 0.21 % / 0.09 % | 1.37 % / 0.03 % |

The reverse engine differentiates the projection itself and stays exact. The
adjoint engines inject the centre profile as their adjoint source, which at
3 % detuning is not exactly the transpose of a projection on the band
profile: their gradient at the ends of the band is as consistent as before
at 1.50 µm and half as consistent at 1.60 µm along the direction. A mode
source that follows the band would remove this.

## Set-up cost

Per port: 5 ms (2-D, 45 nm) and 45 ms (2-D, 20 nm, 172-sample window); for
the 3-D wire window (96 × 68 samples at 20 nm) 0.74 s, against 0.09 s for the
centre solve alone (the helper solves the three modes nearest the profile's
index at each frequency and solves the centre once more to find the solver
that made the profile).
