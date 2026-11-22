"""lightsim.fem against femwell on the same cross-sections (skipped without femwell).

femwell is a test-time reference only, never a dependency of lightsim.
"""

import pytest

pytest.importorskip("femwell")
pytest.importorskip("gmsh")

import fem_femwell_reference as ref  # noqa: E402


def test_heater_temperature_and_index_shift():
    a = ref.heater_femwell()
    b = ref.heater_lightsim(order=2)
    assert abs(b["T_core"] / a["T_core"] - 1) < 3e-3, (a, b)
    assert abs(b["T_max"] / a["T_max"] - 1) < 5e-3, (a, b)
    assert abs(b["dneff"] / a["dneff"] - 1) < 1e-2, (a, b)
    assert abs(b["neff"] - a["neff"]) < 3e-3, (a, b)
