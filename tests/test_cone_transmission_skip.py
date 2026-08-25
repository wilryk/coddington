"""Exactness tests for the transmission-skip optimisation in
:mod:`heliostat.trace.cone`.

A sample whose whole angular footprint provably clears every boundary that
can exist (secondary aperture, receiver window) is exempted from the
``mask_nodes``² transmission raster and deposited with ``frac = 1.0``
directly. Every test here traces the same heliostat twice -- once with the
skip enabled and once with ``cone.DISABLE_TRANSMISSION_SKIP`` forced True,
which routes every sample through the raster regardless -- and requires the
two flux grids to be ``np.array_equal``. Since a raster-measured sample that
happens to be fully transmitted takes the exact same ``mask=None`` path
through :func:`~heliostat.trace.kernels.deposit` as a skipped one (see
``full_pass`` in ``trace_heliostat_cone``), bit-identical output is
equivalent to proving every skipped sample's true transmission actually was
1.0 -- the raster forced back on either reproduces that fact or, if the
bound were wrong somewhere, changes the answer and fails the comparison.

Geometries are chosen so the bound is exercised where it must hold off, not
just where it's free to fire: a receiver window and a cylinder height too
short for a spot to fully clear it, an axicon and a Cassegrain secondary
aperture tight enough that part of a heliostat's own cone genuinely misses
it, and a cylinder receiver whose coordinate seam sits inside the sun cone
(the ``u_period_mm`` case, where u stops being a wall the skip test may
compare against -- see the fix in ``trace_heliostat_cone``). Each clipping
case also asserts on the counters to confirm the raster, not the skip,
did the work there.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from heliostat.geometry.aiming import solve_prime_focus_to_receiver
from heliostat.geometry.receiver import CylinderReceiver, FlatWindowReceiver, FrustumReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace import cone
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from test_mc_parity import _geometry_for, _load_fixture

KERNEL = sunshape_kernel("super_gauss")

# Two field radii from the mc_parity golden fixture: 48 sits at ~35 m (near
# the tower), 574 at ~90 m (the far edge of that fixture's field).
NEAR_HELIOSTAT = 48
FAR_HELIOSTAT = 574
# Low, mid and high sun (solar_el_deg ~= 1.9, 44.9, 79.6) at fixed heliostats.
STEPS = ["20260321_1828", "20260321_0939", "20260321_1235"]


def _row(config: str, heliostat_id: int, step_key: str):
    _, _, summary = _load_fixture(config)
    return summary.loc[(heliostat_id, step_key)]


def _trace_row(row, secondary, receiver, **kwargs) -> dict:
    return trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        **kwargs,
    )


def _assert_matches_skip_forced_off(monkeypatch, tracer, label: str) -> tuple[dict, dict]:
    """Run ``tracer()`` with the skip enabled and again forced off, and
    assert the two traces agree exactly. Returns ``(skip_on, skip_off)``."""
    monkeypatch.setattr(cone, "DISABLE_TRANSMISSION_SKIP", False)
    skip_on = tracer()
    monkeypatch.setattr(cone, "DISABLE_TRANSMISSION_SKIP", True)
    skip_off = tracer()

    assert np.array_equal(skip_on["flux"], skip_off["flux"]), f"{label}: flux grids differ"
    assert skip_on["power_w"] == skip_off["power_w"], f"{label}: power_w differs"
    assert skip_on["incident_power_w"] == skip_off["incident_power_w"], f"{label}: incident power differs"
    c_on, c_off = skip_on["counters"], skip_off["counters"]
    assert c_off["transmission_skipped"] == 0, f"{label}: skip fired while forced off"
    for key in ("samples", "valid", "masked", "blocked", "node_fallback", "unresolved"):
        assert c_on[key] == c_off[key], f"{label}: counter {key} differs ({c_on} vs {c_off})"
    return skip_on, skip_off


# ---------------------------------------------------------------------------
# 1. Broad sweep: every named layout, near/far field position, three sun
#    elevations, both deposit orders -- the general-case proof.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_key", STEPS)
@pytest.mark.parametrize("heliostat_id", [NEAR_HELIOSTAT, FAR_HELIOSTAT])
@pytest.mark.parametrize("config", ["prime_focus", "axicon", "cassegrain"])
def test_flux_bit_identical_across_layouts(monkeypatch, config, heliostat_id, step_key):
    row = _row(config, heliostat_id, step_key)
    secondary, receiver = _geometry_for(config)
    _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_row(row, secondary, receiver, order=1),
        label=f"{config} h{heliostat_id} {step_key} order=1",
    )


@pytest.mark.parametrize("config", ["prime_focus", "axicon", "cassegrain"])
def test_flux_bit_identical_order_2(monkeypatch, config):
    # Order 2 adds the Hessian term to the skip test's reach bound
    # (`_reach_mm`'s `hmax` branch) -- a separate code path from order 1,
    # spot-checked here rather than crossed with every case above.
    row = _row(config, FAR_HELIOSTAT, "20260321_0939")
    secondary, receiver = _geometry_for(config)
    _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_row(row, secondary, receiver, order=2),
        label=f"{config} h{FAR_HELIOSTAT} order=2",
    )


# ---------------------------------------------------------------------------
# 2. Receiver-window edge: a window too small for the spot to fully clear,
#    at both field radii -- the skip must abstain, not approximate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heliostat_id,half_mm", [(NEAR_HELIOSTAT, 550.0), (FAR_HELIOSTAT, 250.0)])
def test_flux_bit_identical_with_window_clipped_spot(monkeypatch, heliostat_id, half_mm):
    row = _row("prime_focus", heliostat_id, "20260321_0939")
    secondary = NoSecondary()
    receiver = FlatWindowReceiver(z_mm=35335.0, half_u_mm=half_mm, half_v_mm=half_mm, facing="down")
    on, _off = _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_row(row, secondary, receiver),
        label=f"window-clipped spot h{heliostat_id}",
    )
    c = on["counters"]
    assert c["masked"] + c["blocked"] + c["node_fallback"] > 0, f"{c}: expected genuine clipping"
    assert c["transmission_skipped"] < c["samples"], f"{c}: skip must not cover a genuinely clipped spot"


# ---------------------------------------------------------------------------
# 3. Secondary rim: an axicon and a Cassegrain aperture tight enough that
#    part of the sun cone genuinely misses it -- requirement 2's "honest
#    secondary-rim case", with both apertures chosen (by direct search) to
#    produce a real mix of skipped and raster-measured samples in one trace.
# ---------------------------------------------------------------------------


def test_flux_bit_identical_with_tight_axicon_aperture(monkeypatch):
    # A far, steep-viewed heliostat (574, ~90 m out) whose outer field misses
    # a tightened aperture: 13500 mm vs. the layout's own 14000 mm default.
    row = _row("axicon", FAR_HELIOSTAT, "20260321_0939")
    secondary, receiver = _geometry_for("axicon")
    tight = dataclasses.replace(secondary, aperture_radius_mm=13500.0)
    on, _off = _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_row(row, tight, receiver),
        label="tight axicon aperture",
    )
    c = on["counters"]
    assert c["transmission_skipped"] > 0, f"{c}: expected some samples clear of the tight rim"
    assert c["masked"] + c["blocked"] + c["node_fallback"] > 0, f"{c}: expected genuine rim clipping"


def test_flux_bit_identical_with_tight_cassegrain_aperture(monkeypatch):
    row = _row("cassegrain", 156, "20260321_0939")
    secondary, receiver = _geometry_for("cassegrain")
    tight = dataclasses.replace(secondary, aperture_radius_mm=10000.0)
    on, _off = _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_row(row, tight, receiver),
        label="tight cassegrain aperture",
    )
    c = on["counters"]
    assert c["transmission_skipped"] > 0, f"{c}: expected some samples clear of the tight rim"
    assert c["masked"] + c["blocked"] + c["node_fallback"] > 0, f"{c}: expected genuine rim clipping"


# ---------------------------------------------------------------------------
# 4. Cylinder receiver: a genuine height (v) edge clip, and the coordinate
#    seam (u_period_mm) -- proof the periodic-u fix changes no output,
#    at bearings that put the reflected image on and off the seam.
# ---------------------------------------------------------------------------

_FIELD_RADIUS_MM = 40000.0
_BASE_SOLAR_AZ_DEG = 165.0
_BASE_SOLAR_EL_DEG = 45.0


def _rotated_case(bearing_deg: float) -> tuple[float, float, float]:
    """Same rigid rotation construction as ``test_receiver_seam.py``: moves
    the heliostat and the sun together around the tower, so a receiver of
    revolution sees a genuinely equivalent scene at every bearing."""
    rad = np.deg2rad(bearing_deg)
    x = _FIELD_RADIUS_MM * np.sin(rad)
    y = _FIELD_RADIUS_MM * np.cos(rad)
    solar_az = (_BASE_SOLAR_AZ_DEG + bearing_deg) % 360.0
    return x, y, solar_az


def _trace_curved(receiver, x_mm, y_mm, solar_az_deg, **kwargs) -> dict:
    sol = solve_prime_focus_to_receiver(x_mm, y_mm, solar_az_deg, _BASE_SOLAR_EL_DEG, receiver)
    return trace_heliostat_cone(
        x_mm,
        y_mm,
        sol.rot_az_deg,
        sol.rot_el_deg,
        sol.c3,
        sol.c4,
        sol.c5,
        solar_az_deg,
        _BASE_SOLAR_EL_DEG,
        NoSecondary(),
        receiver,
        KERNEL,
    )


@pytest.mark.parametrize("receiver_kind", ["cylinder", "frustum"])
@pytest.mark.parametrize("bearing_deg", [0.0, 1.0, 2.0, 5.0, 45.0, 165.0])
def test_flux_bit_identical_near_and_away_from_seam(monkeypatch, receiver_kind, bearing_deg):
    if receiver_kind == "cylinder":
        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
    else:
        receiver = FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0)
    x, y, solar_az = _rotated_case(bearing_deg)
    _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_curved(receiver, x, y, solar_az),
        label=f"{receiver_kind} bearing={bearing_deg}",
    )


def test_flux_bit_identical_with_cylinder_height_clipped(monkeypatch):
    # Height short enough that the spot itself cannot fully clear the v
    # extent, at a bearing well away from the seam so only the (real) height
    # edge is under test.
    receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=650.0)
    x, y, solar_az = _rotated_case(165.0)
    on, _off = _assert_matches_skip_forced_off(
        monkeypatch,
        lambda: _trace_curved(receiver, x, y, solar_az),
        label="cylinder height-clipped",
    )
    c = on["counters"]
    assert c["masked"] + c["blocked"] + c["node_fallback"] > 0, f"{c}: expected genuine height clipping"
    assert c["transmission_skipped"] == 0, (
        f"{c}: every sample here should need the raster -- a nonzero skip would mean "
        "the bound is treating a real clip as clear"
    )
