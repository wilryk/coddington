"""Golden-fixture parity test for the aiming/pointing strategies.

``tests/fixtures/mc_parity/<config>/summary.csv`` carries, for five
heliostats x three sun positions x three optical layouts, both the inputs
(heliostat position, sun position) and the ``solve()`` outputs the private
port traced with (pointing, figure, angle of incidence, cosine efficiency).
Reconstructing the same layout geometry and reproducing every one of those
output columns to ``rtol=1e-9`` is the gate: it says
:mod:`heliostat.geometry.aiming` is not just plausible, it is the same
arithmetic that produced the rays these fixtures store.

Layout parameters below are read off ``LAYOUT_OVERRIDES`` and the
``[geometry]`` defaults in the private repo's ``config.toml`` (recorded here
as plain numbers, the same way ``tests/test_mc_parity.py::_geometry_for``
carries its own geometry forward) -- see the module docstring there for
where each one comes from:

* ``secondary_height_mm = 27000.0``, ``receiver_offset_mm = -20000.0``
  (so ``receiver_height_mm = 7000.0``), ``axicon_angle_deg = 20.0`` are the
  private config's un-overridden ``[geometry]`` defaults, shared by all
  three fixture configs.
* ``prime_focus`` overrides ``focus_height_mm = 35335.0``.
* ``cassegrain`` overrides ``focus_height_mm = 34892.4`` (its own
  ``secondary_rim_height_mm`` override does not affect pointing).
* ``axicon`` overrides ``axicon_aperture_radius_mm``, which
  :func:`~heliostat.geometry.aiming.solve_axicon` never reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from heliostat.geometry.aiming import solve_axicon, solve_cassegrain, solve_prime_focus
from heliostat.geometry.design import Spherical, flower
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
MC_ROOT = FIXTURES_ROOT / "mc_parity"

# The private repo's config.toml [geometry] defaults, un-overridden by any
# of the three fixture configs -- see module docstring.
SECONDARY_HEIGHT_MM = 27000.0
RECEIVER_OFFSET_MM = -20000.0
RECEIVER_HEIGHT_MM = SECONDARY_HEIGHT_MM + RECEIVER_OFFSET_MM  # 7000.0
AXICON_ANGLE_DEG = 20.0

# Per-config focus_height_mm overrides (LAYOUT_OVERRIDES in the private
# scripts/export_public_fixtures.py).
PRIME_FOCUS_HEIGHT_MM = 35335.0
CASSEGRAIN_FOCUS_HEIGHT_MM = 34892.4

RTOL = 1e-9

GATE_COLUMNS = ["rot_az_deg", "rot_el_deg", "c3", "c4", "c5", "aoi_deg", "cosine_efficiency"]


def _solve_row(config: str, row) -> dict:
    if config == "prime_focus":
        sol = solve_prime_focus(
            row.x_mm, row.y_mm, row.solar_az_deg, row.solar_el_deg, PRIME_FOCUS_HEIGHT_MM
        )
    elif config == "cassegrain":
        sol = solve_cassegrain(
            row.x_mm, row.y_mm, row.solar_az_deg, row.solar_el_deg, CASSEGRAIN_FOCUS_HEIGHT_MM
        )
    elif config == "axicon":
        sol = solve_axicon(
            row.x_mm,
            row.y_mm,
            row.solar_az_deg,
            row.solar_el_deg,
            SECONDARY_HEIGHT_MM,
            AXICON_ANGLE_DEG,
            RECEIVER_HEIGHT_MM,
        )
    else:
        raise ValueError(f"unknown config {config!r}")
    return {
        "rot_az_deg": sol.rot_az_deg,
        "rot_el_deg": sol.rot_el_deg,
        "c3": sol.c3,
        "c4": sol.c4,
        "c5": sol.c5,
        "aoi_deg": sol.aoi_deg,
        "cosine_efficiency": sol.cosine_efficiency,
    }


def _cases():
    cases = []
    for config in ("prime_focus", "axicon", "cassegrain"):
        summary = pd.read_csv(MC_ROOT / config / "summary.csv")
        for _, row in summary.iterrows():
            cases.append((config, row))
    return cases


@pytest.mark.parametrize(
    "config,row", _cases(), ids=[f"{c}-{r.heliostat_id}-{r.step_key}" for c, r in _cases()]
)
def test_aiming_matches_fixture(config, row):
    computed = _solve_row(config, row)
    for col in GATE_COLUMNS:
        expected = float(row[col])
        got = float(computed[col])
        assert got == pytest.approx(expected, rel=RTOL, abs=1e-12), (
            f"{config} heliostat={row.heliostat_id} step={row.step_key} column={col}: "
            f"got {got!r}, expected {expected!r}"
        )


def test_aiming_all_45_rows_covered():
    """Sanity on the gate's own shape: 3 configs x 5 heliostats x 3 sun
    positions = 45 rows, matching the module docstring's claim."""
    assert len(_cases()) == 45


def test_canted_design_smoke():
    """Solves are design-agnostic pointing: a flower design traces fine at a
    plain solve's pointing. No assertion beyond sanity (power > 0) -- this
    documents the seam between aiming and shape, it is not a parity gate.
    """
    summary = pd.read_csv(MC_ROOT / "prime_focus" / "summary.csv")
    row = summary.iloc[0]

    sol = solve_prime_focus(
        row.x_mm, row.y_mm, row.solar_az_deg, row.solar_el_deg, PRIME_FOCUS_HEIGHT_MM
    )

    design = flower(
        n_petals=5,
        petal_length_mm=2000.0,
        petal_width_mm=900.0,
        hub_radius_mm=200.0,
        surface=Spherical("slant"),
        cant_focal_mm=95000.0,
        petals_as_facets=True,
    )
    receiver = FlatWindowReceiver(
        z_mm=PRIME_FOCUS_HEIGHT_MM, half_u_mm=2000.0, half_v_mm=2000.0, facing="down"
    )

    out = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        sol.rot_az_deg,
        sol.rot_el_deg,
        sol.c3,
        sol.c4,
        sol.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        NoSecondary(),
        receiver,
        sunshape_kernel("super_gauss"),
        design=design,
    )
    assert out["power_w"] > 0.0
