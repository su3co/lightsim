# SiEPIC fixture

`halfring_gap100nm_r5um.dat` is a copy of
`Lumerical_EBeam_CML/EBeam-dev/source_data/ebeam_dc_halfring_straight_te1550/ebeam_dc_halfring_straight_te1550_gap=100nm_radius=5um_width=500nm_thickness=220nm_CoupleLength=0um.dat`
from the SiEPIC EBeam PDK (github.com/SiEPIC/SiEPIC_EBeam_PDK, commit
363fbb5, MIT licence, copyright Lukas Chrostowski and contributors). It is
used only as a fixture of the INTERCONNECT text format: the tests parse it,
write it back and check its phase convention. No simulation is compared
with it.

The SiEPIC layer numbers lightsim uses (Si 1/0, PinRec 1/10, DevRec 68/0,
Text 10/0) are those of `klayout/EBeam/EBeam.lyp` in the same commit.
