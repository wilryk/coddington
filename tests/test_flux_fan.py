"""v0.2 followups item 2: the frustum's TRUE developed ("fan") view.

Pure-geometry pins for the helpers ``heliostat.web.app._frustum_fan_*``
extract from ``_render_flux_fan_png`` precisely so they can be checked here
without touching matplotlib -- see that function's own docstring for the
cone-development derivation these tests independently verify: unrolling a
right circular cone's lateral surface gives an annular sector whose radius
from the (virtual) apex is the TRUE slant distance and whose angle is the
full azimuth scaled by ``sin(half_angle) = |r_top - r_bot| / slant_length``.

Also pins item 1's azimuth-degree axis helpers (``_azimuth_deg_ticks``,
``_AZIMUTH_CARDINALS``) and item 2's rectangle-view distortion disclosure
(``_frustum_rect_distortion_note``), reusing the exact default frustum
(``PRIME_FOCUS_FRUSTUM_*``) the task's own worked example ("+30% at one rim
and -19% at the other") is stated against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from heliostat.geometry.receiver import FrustumReceiver
from heliostat.web.app import (
    PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM,
    PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM,
    _AZIMUTH_CARDINALS,
    _azimuth_deg_ticks,
    _frustum_fan_cardinal_points_m,
    _frustum_fan_sin_half_angle,
    _frustum_fan_xy_grid_m,
    _frustum_rect_distortion_note,
)

#: The shipped default frustum -- same shape the worked distortion example
#: in the v0.2 followups brief ("+30%/-19%") and PRIME_FOCUS_FRUSTUM_* in
#: app.py itself describe.
DEFAULT_FRUSTUM = FrustumReceiver(
    z_bot_mm=17000.0,
    r_bot_mm=PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM,
    z_top_mm=23000.0,
    r_top_mm=PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM,
)
#: An INVERTED frustum (wide end at the bottom) -- exercises the
#: r_top < r_bot branch every "narrow rim / wide rim" (not "bottom/top")
#: formula above needs to get right.
INVERTED_FRUSTUM = FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0)

GRID = (24, 16)


def _edges(frustum):
    u_edges, v_edges = frustum.bin_edges(GRID)
    return u_edges, v_edges


# ---------------------------------------------------------------------------
# sin(half_angle): one constant, self-consistent from either rim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frustum", [DEFAULT_FRUSTUM, INVERTED_FRUSTUM])
def test_sin_half_angle_matches_both_rims_independently(frustum):
    """sin(half_angle) computed from the bottom-rim triangle must equal the
    SAME ratio computed from the top-rim triangle -- the defining property
    of a right circular cone (r(z)/rho(z) constant at every latitude) that
    makes one sector angle correct for the whole developed band."""
    sin_half_angle = _frustum_fan_sin_half_angle(frustum)
    z_apex = frustum.z_bot_mm - frustum.r_bot_mm * (frustum.z_top_mm - frustum.z_bot_mm) / (
        frustum.r_top_mm - frustum.r_bot_mm
    )
    rho_bot = math.hypot(frustum.z_bot_mm - z_apex, frustum.r_bot_mm)
    rho_top = math.hypot(frustum.z_top_mm - z_apex, frustum.r_top_mm)
    assert frustum.r_bot_mm / rho_bot == pytest.approx(sin_half_angle, rel=1e-9)
    assert frustum.r_top_mm / rho_top == pytest.approx(sin_half_angle, rel=1e-9)
    # |rho_top - rho_bot| must equal the receiver's own slant_length_mm --
    # the two rims are exactly one slant-length apart along the cone's
    # surface, in EITHER direction (a normal frustum's apex sits below the
    # band, rho increasing bot -> top; an inverted one's sits above it, rho
    # decreasing -- see _frustum_fan_rho_direction).
    assert abs(rho_top - rho_bot) == pytest.approx(frustum.slant_length_mm, rel=1e-9)


# ---------------------------------------------------------------------------
# the fan's own radius, at each rim, reproduces r_bot_mm / r_top_mm exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frustum", [DEFAULT_FRUSTUM, INVERTED_FRUSTUM])
def test_fan_radius_at_each_rim_matches_r_bot_and_r_top(frustum):
    """VERIFY-section check: 'the fan view's arc geometry matches
    r_bot_mm/r_top_mm'. Every point on the bottom-rim row of the developed
    grid sits at distance rho_bot from the origin; every point on the
    top-rim row sits at rho_top. rho(v) * sin_half_angle == r(v) is the
    algebraic identity that makes this exact at every v, not just the two
    rims -- checked directly here at the two rims (the physically
    meaningful ones) plus a random interior v."""
    u_edges, v_edges = _edges(frustum)
    x_grid_m, y_grid_m = _frustum_fan_xy_grid_m(frustum, u_edges, v_edges)
    sin_half_angle = _frustum_fan_sin_half_angle(frustum)

    rho_bot_row_m = np.hypot(x_grid_m[0], y_grid_m[0])
    rho_top_row_m = np.hypot(x_grid_m[-1], y_grid_m[-1])
    assert np.allclose(rho_bot_row_m * sin_half_angle * 1000.0, frustum.r_bot_mm, rtol=1e-9)
    assert np.allclose(rho_top_row_m * sin_half_angle * 1000.0, frustum.r_top_mm, rtol=1e-9)

    # A random interior row: r(v) linearly interpolated between the rims.
    mid_row = len(v_edges) // 3
    v_mm = v_edges[mid_row]
    frac = v_mm / frustum.slant_length_mm
    expected_r_mm = frustum.r_bot_mm + frac * (frustum.r_top_mm - frustum.r_bot_mm)
    rho_row_m = np.hypot(x_grid_m[mid_row], y_grid_m[mid_row])
    assert np.allclose(rho_row_m * sin_half_angle * 1000.0, expected_r_mm, rtol=1e-9)


def test_fan_grid_shape_matches_bin_edges():
    """(X, Y) must be one row/col larger than the flux array in each
    direction, exactly like _render_flux_png's own imshow extent -- the
    same (n_v+1, n_u+1) corner-grid convention pcolormesh(shading='flat')
    needs relative to an (n_v, n_u) flux array."""
    u_edges, v_edges = _edges(DEFAULT_FRUSTUM)
    x_grid_m, y_grid_m = _frustum_fan_xy_grid_m(DEFAULT_FRUSTUM, u_edges, v_edges)
    n_u, n_v = GRID
    assert x_grid_m.shape == (n_v + 1, n_u + 1)
    assert y_grid_m.shape == (n_v + 1, n_u + 1)


# ---------------------------------------------------------------------------
# cardinal markers: correct bearing, both N's distinct points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frustum", [DEFAULT_FRUSTUM, INVERTED_FRUSTUM])
def test_cardinal_points_sit_at_the_correct_scaled_bearing(frustum):
    """Each cardinal's angular position must be its u-fraction (from
    _AZIMUTH_CARDINALS: S=0, E/W=+-1/2, N=+-1 of the half-circumference)
    scaled through the SAME sin_half_angle the flux surface itself uses --
    independently recomputed here from raw u_edges/r_mean_mm rather than by
    calling the helper's own internals twice."""
    u_edges, v_edges = _edges(frustum)
    sin_half_angle = _frustum_fan_sin_half_angle(frustum)
    rho_bot_mm = frustum.r_bot_mm / sin_half_angle
    # Independent of _frustum_fan_rho_direction: the OUTER (apex-farthest)
    # rim is the bottom for an inverted frustum (r_top < r_bot), the top
    # otherwise -- computed here straight from r_top_mm/r_bot_mm, not by
    # calling that helper.
    direction = 1.0 if frustum.r_top_mm >= frustum.r_bot_mm else -1.0
    rho_top_mm = rho_bot_mm + direction * float(v_edges[-1])
    rho_out_m = max(rho_bot_mm, rho_top_mm) * 1.05 / 1000.0
    half_circ_u = 0.5 * (float(u_edges[-1]) - float(u_edges[0]))

    points = _frustum_fan_cardinal_points_m(frustum, u_edges, v_edges)
    assert len(points) == len(_AZIMUTH_CARDINALS) == 5
    for (letter, frac), (got_letter, x_m, y_m) in zip(_AZIMUTH_CARDINALS, points):
        assert got_letter == letter
        phi = (frac * half_circ_u / frustum.r_mean_mm) * sin_half_angle
        assert x_m == pytest.approx(rho_out_m * math.sin(phi), rel=1e-9)
        assert y_m == pytest.approx(-rho_out_m * math.cos(phi), rel=1e-9)

    # S (frac=0) sits on the vertical centreline; both N's are the same
    # physical seam (same rho from the origin) but on OPPOSITE sides --
    # distinct points, not a single one silently overwritten.
    s_letter, s_x, s_y = points[2]
    assert s_letter == "S"
    assert s_x == pytest.approx(0.0, abs=1e-9)
    n_left = points[0]
    n_right = points[-1]
    assert n_left[0] == "N" and n_right[0] == "N"
    assert n_left[1] != pytest.approx(n_right[1])  # different x -- two distinct points
    assert math.hypot(n_left[1], n_left[2]) == pytest.approx(math.hypot(n_right[1], n_right[2]), rel=1e-9)


# ---------------------------------------------------------------------------
# item 1: azimuth-degree tick helper, shares _AZIMUTH_CARDINALS with the fan
# ---------------------------------------------------------------------------


def test_azimuth_deg_ticks_positions_and_labels():
    u_period_mm = 2.0 * math.pi * 3000.0  # a plausible cylinder circumference
    positions, labels = _azimuth_deg_ticks(u_period_mm)
    assert positions == pytest.approx([-u_period_mm / 2, -u_period_mm / 4, 0.0, u_period_mm / 4, u_period_mm / 2])
    assert labels == ["N\n-180°", "W\n-90°", "S\n0°", "E\n90°", "N\n180°"]


# ---------------------------------------------------------------------------
# item 2: rectangle-view distortion disclosure matches the worked example
# ---------------------------------------------------------------------------


def test_rect_distortion_note_matches_the_worked_example():
    """The task's own worked numbers for the shipped default frustum:
    r_bot=2500, r_top=4000, r_mean=3250 -> narrow rim (bottom, 2500)
    stretched +30%, wide rim (top, 4000) compressed -19%."""
    note = _frustum_rect_distortion_note(DEFAULT_FRUSTUM)
    assert "+30%" in note
    assert "-19%" in note
    assert "narrow rim" in note and "wide rim" in note


def test_rect_distortion_note_flips_correctly_for_an_inverted_frustum():
    """Same physical distortion, but the geometric NARROW end is now the
    TOP (r_top=2500) and the WIDE end is the BOTTOM (r_bot=4000) -- the
    note must still say "narrow"/"wide" correctly rather than "bottom"/
    "top", which would flip sign silently."""
    note = _frustum_rect_distortion_note(INVERTED_FRUSTUM)
    assert "+30%" in note  # the narrow rim (r=2500) is still stretched +30%
    assert "-19%" in note  # the wide rim (r=4000) is still compressed -19%
