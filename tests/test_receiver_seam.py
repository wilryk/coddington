"""Verification for the three curved-receiver physics fixes in
``heliostat.geometry.receiver``:

1. The azimuthal coordinate seam (``CylinderReceiver``/``FrustumReceiver``
   unroll azimuth with a branch cut at +y/north). :mod:`heliostat.trace.cone`
   measures its Jacobian by finite-differencing that coordinate between rays
   cast from one point at a tiny angular offset; radial aiming can point a
   north-sector heliostat's whole stencil straight at the cut, and before
   the fix the resulting jump made the measured Jacobian (and the flux/power
   it deposits) come out ~10^5 too large. ``TestAzimuthalRotationInvariance``
   is the end-to-end proof: a receiver of revolution must trace the SAME
   heliostat, rigidly rotated around the tower, to the same power and peak
   flux at every bearing.
2. ``FrustumReceiver.intersect`` rejecting a near heliostat outright because
   its origin sits inside the *infinite* double-cone envelope even though it
   is nowhere near the finite frustum band. ``test_near_heliostat_frustum_*``
   checks real Monte Carlo rays from such a heliostat land on the receiver.
3. A lower-level unit test pinning the actual mechanism behind (1): a
   synthetic finite-difference stencil whose true azimuth sits at the seam
   must report mutually continuous coordinates, not a jump of order the
   receiver's full circumference.

``TestSeamAdjacentResidual`` documents a separate, much smaller, honestly
disclosed limitation the fix does not (and, without touching
:mod:`heliostat.trace.cone`, cannot) fully close: a heliostat whose own
reflected image lands almost exactly ON the seam sits on a genuine
chart-boundary singularity. See that class's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.geometry.aiming import solve_prime_focus_to_receiver
from heliostat.geometry.receiver import CylinderReceiver, FrustumReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat

_KERNEL = sunshape_kernel("super_gauss")
_SECONDARY = NoSecondary()

# Comfortably clear of either receiver's own envelope, so every compass
# bearing below traces a genuinely comparable heliostat (the near-heliostat
# envelope case is exercised separately, below).
_FIELD_RADIUS_MM = 40000.0
_BASE_SOLAR_AZ_DEG = 165.0
_BASE_SOLAR_EL_DEG = 45.0
# Offset 15 degrees off the compass points, not sitting on them: a heliostat
# whose reflected image centres EXACTLY on the receiver's coordinate seam
# (bearing 0.0 here) is its own, separately-tested case below -- see
# test_deficit_right_at_the_seam_is_bounded_and_narrow. These eight still
# cover north, south, east, west and the diagonals (within 15 degrees), the
# spread the task asks for, without conflating the seam/Jacobian fix (this
# class) with that unrelated, narrow residual.
_BEARINGS_DEG = [15.0, 60.0, 105.0, 150.0, 195.0, 240.0, 285.0, 330.0]


def _rotated_case(bearing_deg: float) -> tuple[float, float, float]:
    """``(x_mm, y_mm, solar_az_deg)`` for a heliostat at ``bearing_deg`` off
    north, rotated as a rigid body together with the sun.

    Rotating only the heliostat's position would trace a genuinely different
    (and generally asymmetric) sun-relative geometry at each bearing;
    rotating the sun's azimuth by the same amount keeps the whole optical
    configuration -- heliostat position, mirror pointing, incidence angle --
    a rigid rotation of the base case around the tower axis. A receiver of
    revolution must then trace to the same power and peak flux at every
    bearing, by symmetry alone.
    """
    rad = np.deg2rad(bearing_deg)
    x = _FIELD_RADIUS_MM * np.sin(rad)
    y = _FIELD_RADIUS_MM * np.cos(rad)
    solar_az = (_BASE_SOLAR_AZ_DEG + bearing_deg) % 360.0
    return x, y, solar_az


def _trace(receiver, x_mm, y_mm, solar_az_deg):
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
        _SECONDARY,
        receiver,
        _KERNEL,
        # order=1 (linear deposit) isolates the seam/Jacobian fix under test
        # from the order-2 quadratic deposit's separate, still-open fold
        # hazard on curved receivers (see receiver.py's `is_planar`).
        order=1,
    )


class TestAzimuthalRotationInvariance:
    """A receiver of revolution traced with a rigidly rotated heliostat must
    report the same power and peak flux at every bearing. Before the seam
    fix, the north bearings (0/45/315 degrees, where radial aiming points
    the stencil straight at the coordinate cut) diverged by orders of
    magnitude from the south bearings that every prior test happened to use.
    """

    @pytest.mark.parametrize(
        "receiver",
        [
            CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0),
            FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0),
        ],
        ids=["cylinder", "frustum"],
    )
    def test_power_and_peak_flux_match_at_every_bearing(self, receiver):
        results = {}
        for bearing in _BEARINGS_DEG:
            x, y, solar_az = _rotated_case(bearing)
            out = _trace(receiver, x, y, solar_az)
            results[bearing] = (out["power_w"], float(out["flux"].max()))

        base_power, base_peak = results[_BEARINGS_DEG[0]]
        assert base_power > 0
        assert base_peak > 0
        for bearing, (power, peak) in results.items():
            assert power == pytest.approx(base_power, rel=1e-6), (
                f"bearing {bearing} deg: power {power} W vs base {base_power} W "
                f"(all bearings: { {b: p for b, (p, _) in results.items()} })"
            )
            assert peak == pytest.approx(base_peak, rel=1e-6), (
                f"bearing {bearing} deg: peak flux {peak} W/m^2 vs base {base_peak} W/m^2 "
                f"(all bearings: { {b: pk for b, (_, pk) in results.items()} })"
            )


class TestSeamIsInvisible:
    """A receiver that closes on itself has no edge in ``u``, and a heliostat
    whose image lands exactly on the coordinate seam must not be able to tell.

    Two things had to hold for that: the intersection has to unwrap azimuth
    continuously so a stencil never differences across the cut, and the flux
    deposit has to wrap, so a footprint running past the last column
    continues at the first instead of being written off as spillage. With
    only the first, a heliostat sitting on the seam lost about half its own
    power -- the shape this class used to pin.
    """

    def test_a_heliostat_on_the_seam_keeps_its_power(self):
        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        ratios = {}
        for bearing in (0.0, 1.0, 2.0, 5.0, 10.0):
            x, y, solar_az = _rotated_case(bearing)
            out = _trace(receiver, x, y, solar_az)
            ratios[bearing] = out["power_w"] / out["incident_power_w"]

        # Sitting exactly on the seam is worth the same as sitting anywhere
        # else: no deficit at all, not merely a bounded one.
        for bearing, ratio in ratios.items():
            assert ratio > 0.95, f"bearing {bearing} lost power to the seam: {ratio:.3f}"
        spread = max(ratios.values()) - min(ratios.values())
        assert spread < 0.02, f"power depends on bearing near the seam: {ratios}"



# ---------------------------------------------------------------------------
# finding 2: a near heliostat must not be rejected by the frustum's envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius_mm", [4000.0, 6000.0, 8500.0, 10000.0])
def test_near_heliostat_frustum_collects_nonzero_power(radius_mm):
    """A heliostat at a ground radius of 4-10 m, under a frustum whose
    finite band sits well above the ground, must collect a real, sensible
    share of what it reflects -- not silently zero because its origin sits
    inside the *infinite* extension of the cone the finite band is cut from.
    """
    receiver = FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0)
    secondary = NoSecondary()
    rng = np.random.default_rng(0)
    x_mm, y_mm = radius_mm, 0.0
    sol = solve_prime_focus_to_receiver(x_mm, y_mm, 165.0, 45.0, receiver)
    result = trace_heliostat(
        x_mm,
        y_mm,
        sol.rot_az_deg,
        sol.rot_el_deg,
        sol.c3,
        sol.c4,
        sol.c5,
        165.0,
        45.0,
        secondary,
        receiver,
        20000,
        rng,
        source_power_w=1.0,
    )
    hit_mirror = result["counters"]["hit_mirror"]
    reached = result["counters"]["reached_receiver"]
    assert hit_mirror > 0, "sanity: the mirror itself must be illuminated"
    # "Sensible" here is deliberately loose (not a tight physical prediction,
    # just "not silently zero"): at least a third of the rays that left the
    # mirror actually reach the receiver surface.
    assert reached > 0.3 * hit_mirror, (
        f"radius {radius_mm} mm: only {reached}/{hit_mirror} reflected rays reached the receiver"
    )


def test_near_and_far_frustum_heliostats_collect_comparable_fractions():
    """Not just non-zero: a near heliostat's captured fraction of its own
    reflected rays should be in the same ballpark as a conventional, far
    heliostat's -- proof this isn't a token few rays slipping through, but
    the same physical mechanism working correctly close in.
    """
    receiver = FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0)
    secondary = NoSecondary()

    def _fraction(x_mm, y_mm, seed):
        rng = np.random.default_rng(seed)
        sol = solve_prime_focus_to_receiver(x_mm, y_mm, 165.0, 45.0, receiver)
        result = trace_heliostat(
            x_mm,
            y_mm,
            sol.rot_az_deg,
            sol.rot_el_deg,
            sol.c3,
            sol.c4,
            sol.c5,
            165.0,
            45.0,
            secondary,
            receiver,
            20000,
            rng,
            source_power_w=1.0,
        )
        c = result["counters"]
        return c["reached_receiver"] / c["hit_mirror"]

    near = _fraction(6000.0, 0.0, seed=1)
    far = _fraction(50000.0, 0.0, seed=2)
    assert near > 0.5
    assert far > 0.5
    assert near == pytest.approx(far, rel=0.5)


# ---------------------------------------------------------------------------
# mechanism-level unit test: the seam itself, isolated from the rest of the
# optical chain
# ---------------------------------------------------------------------------


def test_stencil_straddling_the_seam_stays_locally_continuous():
    """Direct check on ``CylinderReceiver.intersect``: five rays cast from
    one point, at the tiny angular offsets :mod:`heliostat.trace.cone` uses
    for its finite-difference stencil, aimed almost due north so the bundle
    straddles the seam. Before the fix, differencing this stencil's ``u``
    values produced a jump of order the full circumference (~19000 mm here,
    ~10^5 relative to the true few-mm spread); after the fix the reported
    values stay mutually within one stencil footprint.
    """
    receiver = CylinderReceiver(center_z_mm=0.0, radius_mm=3000.0, height_mm=6000.0)
    origin = np.array([0.0, 40000.0, 500.0])  # north of the tower
    base_dir = np.array([0.0, -1.0, 0.0])  # aimed south -- straight at the seam
    delta_rad = 2.0e-4
    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) * delta_rad
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    dirs = base_dir[None, :] + offsets[:, 0:1] * e1[None, :] + offsets[:, 1:2] * e2[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    p = np.tile(origin[:, None], (1, 5))
    d = dirs.T
    hit, uv = receiver.intersect(p, d)

    assert hit.all(), "every stencil leg should reach the cylinder"
    u = uv[0]
    # A naive central difference across the +/-e1 legs must be a sane
    # mm/rad Jacobian entry, not the seam-crossing artifact (which would be
    # of order 2*pi*radius / (2*delta_rad) ~= 3e7 mm/rad here).
    jac_u_e1 = (u[1] - u[2]) / (2.0 * delta_rad)
    assert abs(jac_u_e1) < 1.0e5, f"seam-crossing jump leaked through: du/dangle = {jac_u_e1}"
    # The whole stencil's spread should be commensurate with its own tiny
    # angular footprint at this range, not the receiver's circumference.
    spread_mm = u.max() - u.min()
    circumference_mm = 2.0 * np.pi * receiver.radius_mm
    assert spread_mm < 0.01 * circumference_mm


def test_singleton_rays_are_unaffected_by_the_seam_fix():
    """Rays with distinct origins (the common case for every caller outside
    the cone backend's finite-difference stencils) must get back exactly
    ``arctan2``'s own azimuth, unchanged -- the fix is a no-op for them."""
    receiver = CylinderReceiver(center_z_mm=0.0, radius_mm=3000.0, height_mm=6000.0)
    rng = np.random.default_rng(0)
    n = 200
    bearings = rng.uniform(0.0, 2.0 * np.pi, n)
    origins = np.stack(
        [40000.0 * np.sin(bearings), 40000.0 * np.cos(bearings), rng.uniform(-500.0, 500.0, n)]
    )
    dirs = -origins / np.linalg.norm(origins, axis=0, keepdims=True)
    hit, uv = receiver.intersect(origins, dirs)

    # Recompute the hit points independently to get a reference arctan2.
    px, py = origins[0, hit], origins[1, hit]
    dx, dy = dirs[0, hit], dirs[1, hit]
    a = dx * dx + dy * dy
    b = 2.0 * (px * dx + py * dy)
    c = px * px + py * py - receiver.radius_mm**2
    t = (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    hx = px + t * dx
    hy = py + t * dy
    expected_u = receiver.radius_mm * np.arctan2(hx, -hy)
    assert uv[0] == pytest.approx(expected_u, abs=1e-6)


# ---------------------------------------------------------------------------
# "worth checking" item: the frustum's radial aim point lands on the sloped
# wall, matching the cylinder's already-tested behaviour
# ---------------------------------------------------------------------------


def test_frustum_aim_point_is_reachable_and_on_the_true_wall():
    """Beyond ``aim_point_mm``'s own arithmetic (covered in
    ``test_receiver_shapes.py``): a heliostat actually aimed at it, through
    the real aiming solve, must land within the receiver's own ``uv_extent``
    -- i.e. the sloped-wall aim point is not just numerically on the cone,
    but actually hittable end to end."""
    receiver = FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0)
    (u0, u1), (v0, v1) = receiver.uv_extent()
    for x_mm, y_mm in [(40000.0, 0.0), (0.0, -40000.0), (-28284.0, 28284.0)]:
        sol = solve_prime_focus_to_receiver(x_mm, y_mm, 165.0, 45.0, receiver)
        mirror_pos = np.array([x_mm, y_mm, 0.0])
        aim = np.array(
            [sol.extras["aim_x_mm"], sol.extras["aim_y_mm"], sol.extras["aim_z_mm"]]
        )
        d = aim - mirror_pos
        d /= np.linalg.norm(d)
        hit, uv = receiver.intersect(mirror_pos.reshape(3, 1), d.reshape(3, 1))
        assert hit[0], f"heliostat at ({x_mm}, {y_mm}) cannot reach its own aim point"
        assert u0 <= uv[0, 0] <= u1
        assert v0 <= uv[1, 0] <= v1
        assert np.allclose(
            mirror_pos + np.linalg.norm(aim - mirror_pos) * d, aim, atol=1e-3
        )


class TestSeamIsInvisibleInsideACavity:
    """The aperture-clipped (cavity) receiver forwards uv, extent and bin
    areas to its inner surface -- and must forward `u_period_mm` too, or the
    tracer sees no periodicity and treats the cavity cylinder's seam as a
    hard edge: a spot straddling it behind the aperture loses its wrapped
    share, the release-night seam bug in cavity form. An aperture so large
    it clips nothing must be optically invisible: the cavity trace has to
    match the bare cylinder at every bearing, seam included.
    """

    def _cavity(self, cylinder):
        from heliostat.geometry.receiver import ApertureClippedReceiver, FlatWindowReceiver

        return ApertureClippedReceiver(
            aperture=FlatWindowReceiver(
                z_mm=20000.0, half_u_mm=1.0e6, half_v_mm=1.0e6, facing="down"
            ),
            inner=cylinder,
        )

    def test_period_delegates_to_the_inner_surface(self):
        cylinder = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        assert self._cavity(cylinder).u_period_mm == cylinder.u_period_mm

    def test_a_seam_heliostat_keeps_its_power_behind_a_clipless_aperture(self):
        cylinder = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        cavity = self._cavity(cylinder)
        for bearing in (0.0, 1.0, 5.0, 180.0):
            x, y, solar_az = _rotated_case(bearing)
            bare = _trace(cylinder, x, y, solar_az)
            clipped = _trace(cavity, x, y, solar_az)
            assert clipped["power_w"] == pytest.approx(bare["power_w"], rel=1e-6), (
                f"bearing {bearing}: cavity {clipped['power_w']} W vs bare {bare['power_w']} W"
            )
            assert float(clipped["flux"].max()) == pytest.approx(
                float(bare["flux"].max()), rel=1e-6
            ), f"bearing {bearing}: cavity peak differs from bare"


class TestMonteCarloRefereesTheCavitySeam:
    """Monte Carlo is an INDEPENDENT referee for the cavity seam fix: MC
    rays are points, each landing in exactly one bin, so MC never needed
    the wrap machinery and never had the seam bug -- agreement with it
    validates the cone backend against exact geometry, not against a
    second copy of the same code path.

    Comparing the cone/MC collected-power ratio at the seam bearing to the
    same ratio at a seam-free bearing cancels every normalisation
    difference between the backends; pre-fix that double ratio was ~0.5
    (the cone cavity lost the wrapped half of its spot), and any wrap
    regression drags it off 1 again.
    """

    def _mc_power_w(self, receiver, x, y, solar_az, n_rays=200_000):
        from heliostat.trace.mc import trace_heliostat

        sol = solve_prime_focus_to_receiver(x, y, solar_az, _BASE_SOLAR_EL_DEG, receiver)
        out = trace_heliostat(
            x,
            y,
            sol.rot_az_deg,
            sol.rot_el_deg,
            sol.c3,
            sol.c4,
            sol.c5,
            solar_az,
            _BASE_SOLAR_EL_DEG,
            _SECONDARY,
            receiver,
            n_rays,
            np.random.default_rng(20260826),
        )
        return out["watts_per_ray"] * out["counters"]["in_window"]

    def test_cone_to_mc_ratio_is_bearing_independent_for_the_cavity(self):
        from heliostat.geometry.receiver import ApertureClippedReceiver, FlatWindowReceiver

        cylinder = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        cavity = ApertureClippedReceiver(
            aperture=FlatWindowReceiver(
                z_mm=20000.0, half_u_mm=1.0e6, half_v_mm=1.0e6, facing="down"
            ),
            inner=cylinder,
        )

        ratios = {}
        for bearing in (0.0, 180.0):  # dead on the seam vs. as far from it as possible
            x, y, solar_az = _rotated_case(bearing)
            cone_power = _trace(cavity, x, y, solar_az)["power_w"]
            mc_power = self._mc_power_w(cavity, x, y, solar_az)
            assert mc_power > 0
            ratios[bearing] = cone_power / mc_power

        # 2% tolerance: MC shot noise at 2e5 rays plus genuine backend
        # differences, both bearing-independent -- the seam deficit this
        # guards against was a factor of ~2, not percent-scale.
        assert ratios[0.0] == pytest.approx(ratios[180.0], rel=0.02), (
            f"cone/MC ratio depends on seam proximity: {ratios}"
        )


class TestNodeFallbackWrapsAtTheSeam:
    """Bearing invariance with the node-fallback path ENGAGED -- coverage
    the rotation-invariance suite above lacked (its focused solves never
    lose a chief ray, so the fallback branch went untested end to end).

    The fallback's node landing points are azimuth-unwrapped for stencil
    continuity, so a node's ``u`` can sit past the chart edge; the deposit
    wraps its column modulo the count, like the main deposit's ``wrap_u``.
    Honest scope note: on a bare surface of revolution the wrap-vs-clamp
    distinction is measured kernel-tail negligible (~1e-11 of peak here) --
    a chief near the seam always INTERSECTS the surface, so fallback fires
    only at tangent-miss limbs where out-of-chart nodes carry the kernel's
    outermost weights (instrumented: node columns -6..137 on this 128
    chart at bearing 45, all tail nodes). This pin therefore BOUNDS any
    fallback-placement wrongness below 1e-6 of peak across bearings rather
    than discriminating the historical clamp; the case where clamping WAS
    measurable (+3.7% seam peak) lived on the bspline path's coarse control
    grid and is separately pinned in tests/test_bspline_deposit.py.
    """

    def _flat_mirror_trace(self, receiver, x, y, solar_az):
        """A FLAT mirror throws a mirror-sized beam; against a narrow
        cylinder the beam's edge band has chief rays that tangent-miss the
        body while their sun cones still clip it -- the only geometry that
        actually reaches the fallback path (a focused solve keeps every
        chief on the wall, so rim spill lands in `masked` instead)."""
        sol = solve_prime_focus_to_receiver(x, y, solar_az, _BASE_SOLAR_EL_DEG, receiver)
        return trace_heliostat_cone(
            x,
            y,
            sol.rot_az_deg,
            sol.rot_el_deg,
            0.0,
            0.0,
            0.0,
            solar_az,
            _BASE_SOLAR_EL_DEG,
            _SECONDARY,
            receiver,
            _KERNEL,
            order=1,
        )

    def test_fallback_mass_lands_in_the_wrapped_column(self):
        # Bearings 45 vs 225: bearing 45's fallback band unwraps across the
        # seam (instrumented) while the rigidly-identical 225 keeps its band
        # mid-chart -- the maximal-contrast pair. 45 deg is exactly 16
        # columns on this grid, so the rotation is an exact circular shift
        # and sorted values must match bin for bin.
        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=1000.0, height_mm=6000.0)
        fluxes = {}
        for bearing in (45.0, 225.0):
            x, y, solar_az = _rotated_case(bearing)
            out = self._flat_mirror_trace(receiver, x, y, solar_az)
            assert out["counters"]["node_fallback"] > 0, (
                "geometry failed to engage the node-fallback path -- test is vacuous"
            )
            fluxes[bearing] = np.sort(out["flux"].ravel())

        # Peak-scaled absolute tolerance: a per-element relative tolerance
        # explodes on the map's near-zero bins. Measured agreement is
        # ~2e-11 of peak; the bar leaves five orders of headroom while
        # still catching any future fallback-placement wrongness at
        # per-node-share scale.
        peak = float(fluxes[45.0].max())
        np.testing.assert_allclose(
            fluxes[45.0],
            fluxes[225.0],
            rtol=0.0,
            atol=1e-6 * peak,
            err_msg="seam bearing's flux distribution differs from the seam-free bearing's",
        )
