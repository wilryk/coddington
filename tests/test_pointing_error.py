"""Pin tests for docs/ui-spec-v0.2.md §F: pointing error.

Convention (resolved 2026-08-25, signed off): the quoted ``pointing_error_mrad``
is the RMS angular deviation of the REFLECTED BEAM -- no factor of two on
reflection is applied to the user's number (unlike ``slope_error_mrad``,
which IS the mirror-tilt RMS and picks the factor of two up from the
reflection law itself). Monte Carlo (``heliostat.trace.mc.trace_heliostat``)
draws ONE shared 2-axis Gaussian mirror-tilt offset per call -- one
heliostat, one instant -- at HALF the quoted beam figure per axis, which
the reflection law's own doubling brings back to exactly the quoted number
in the realised beam. Cone modes (``heliostat.trace.cone.sunshape_kernel``)
fold the same ``pointing_error_mrad`` in as an added broadening term with NO
doubling of its own -- see both functions' docstrings for the full
bookkeeping.

Four things are pinned here, matching the spec's own acceptance gate:

1. Zero pointing error is bit-identical, both backends.
2. The MC statistical pin: at a fixed seed, over many independent
   timesteps, the realised reflected-beam angular deviation RMS equals the
   quoted mrad, measured from the actual receiver spot displacement (never
   from the internal random draw).
3. Cone-vs-MC ensemble parity: cone's broadened kernel matches the MC
   ENSEMBLE (averaged over many timestep draws) spot RMS, while a SINGLE MC
   instant legitimately does not (it is a shift, not a spread) -- that
   asymmetry is the spec's own design, documented rather than hidden.
4. Quasi-static reproducibility: same seed (+ timestep) gives the same
   offset; a different one gives a different offset.

Geometry is borrowed from ``test_mc_physics_fixes``'s specularity fixture
(``_reflect``, ``_specularity_only_receiver``, the 45-deg-incidence
constants) -- the same sibling-import pattern ``test_cone_vs_mc.py`` and
``test_cone_convergence.py`` already use (no ``__init__.py`` under
``tests/``, so pytest puts it on ``sys.path``), rather than duplicating a
geometry helper a third time. That fixture's own receiver is already sized
generously (6 m half-width against a sub-metre spot) and centred exactly on
the unperturbed chief ray, which is exactly what a displacement-from-chief
measurement needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat
from heliostat.trace.samplers import SuperGaussSampler
from test_mc_parity import _geometry_for, _load_fixture
from test_mc_physics_fixes import (
    SPEC_ROT_AZ,
    SPEC_ROT_EL,
    SPEC_SUN_AZ,
    SPEC_SUN_EL,
    _reflect,
    _specularity_only_receiver,
)

N_RAYS = 4000

# A SECOND, near-normal-incidence geometry, distinct from test_mc_physics_
# fixes' 45-deg-incidence SPEC_* constants above. Tilting the mirror's NORMAL
# (what slope_error_mrad and pointing_error_mrad both perturb, pre-reflection
# -- unlike specularity_mrad, which perturbs the outgoing ray directly) is
# only isotropic in the OUTGOING beam's own tangential/sagittal frame AT
# NORMAL INCIDENCE: a first-order derivation of the reflection law shows a
# tangent-plane normal tilt of magnitude delta deflects the beam by exactly
# `2*delta` in the tangential (in-plane) sense but only `2*cos(theta_i)*delta`
# in the sagittal sense -- a real, pre-existing geometric property shared by
# slope_error_mrad (never previously stress-tested at fixed oblique
# incidence; only specularity_mrad's own, DIFFERENT post-reflection isotropy
# was), not a bug introduced here. The `hypot(...)` isotropic broadening both
# backends fold errors in with is itself the same "isotropic in the mirror's
# own frame" approximation, so exercising it at 45-deg incidence would
# conflate that pre-existing obliquity effect with the specific
# doubling/halving bookkeeping this module pins. 3-degree incidence
# (cos(3deg) = 0.9986) keeps that conflation under 0.2%, negligible next to
# this module's tolerances, while staying off the exactly-zero incidence
# singularity where the tangential/sagittal axes below are undefined.
NORMAL_ROT_AZ, NORMAL_ROT_EL = 0.0, 88.0
NORMAL_SUN_AZ, NORMAL_SUN_EL = 90.0, 85.0


def _near_normal_receiver():
    """A FlatWindowReceiver centred on the near-normal-incidence chief ray,
    the same construction as ``test_mc_physics_fixes._specularity_only_receiver``
    generalised to this module's own geometry."""
    _, _, _, _, d_out = _reflect(NORMAL_ROT_AZ, NORMAL_ROT_EL, NORMAL_SUN_AZ, NORMAL_SUN_EL)
    z_mm = 30_000.0
    t = z_mm / d_out[2]
    cx, cy = d_out[0] * t, d_out[1] * t
    return FlatWindowReceiver(
        z_mm=z_mm, half_u_mm=6000.0, half_v_mm=6000.0, facing="down",
        center_x_mm=cx, center_y_mm=cy,
    )


def _mean_tang_sag_deviation(
    pointing_error_mrad: float,
    rng: np.random.Generator,
    rot_az_deg: float,
    rot_el_deg: float,
    sun_az_deg: float,
    sun_el_deg: float,
    receiver,
    n_rays: int = N_RAYS,
):
    """Trace once and return (tang_rad, sag_rad, n_landed): the MEAN
    reflected-ray deviation from the exact, unperturbed chief direction
    ``d_out``, averaged over every ray that reached the receiver window in
    THIS ONE call -- i.e. this instant's own spot centroid, in angle,
    reconstructed exactly from ``paths`` (mirror hit -> receiver hit is a
    straight line with no secondary), the same non-tautological
    reconstruction ``test_mc_physics_fixes._tangential_sagittal_rms`` uses
    for specularity. Averaging over many rays within one instant cancels
    the sun's own angular width (each ray's individual deviation) down to a
    small residual, isolating the ONE shared pointing offset this instant
    drew -- exactly the ensemble/instant distinction the spec draws (§F).
    """
    n, u, v, d_in, d_out = _reflect(rot_az_deg, rot_el_deg, sun_az_deg, sun_el_deg)
    secondary = NoSecondary()
    out = trace_heliostat(
        0.0, 0.0, rot_az_deg, rot_el_deg, 0.0, 0.0, 0.0, sun_az_deg, sun_el_deg,
        secondary, receiver, n_rays, rng,
        pointing_error_mrad=pointing_error_mrad, return_paths=True,
    )
    paths = out["paths"]
    mir, rec = paths[1], paths[3]
    n_landed = mir.shape[1]
    if n_landed == 0:
        return 0.0, 0.0, 0
    dvec = rec - mir
    dvec /= np.linalg.norm(dvec, axis=0)

    t_hat = n - np.dot(n, d_out) * d_out
    t_hat /= np.linalg.norm(t_hat)
    q_hat = np.cross(d_out, t_hat)

    delta = dvec - d_out[:, None]
    tang = float(np.mean(delta.T @ t_hat))
    sag = float(np.mean(delta.T @ q_hat))
    return tang, sag, n_landed


def _cone_centroid_rms_mm(cone: dict):
    """(centroid_uv_mm, rms_mm) from a cone backend flux grid -- the same
    recipe ``heliostat.web.app._cone_metrics``/``test_cone_convergence``'s
    ``_centroid_rms`` use, kept local so this module stays independent of
    ``heliostat.web.app``."""
    flux = cone["flux"]
    u_mid = 0.5 * (cone["u_edges"][:-1] + cone["u_edges"][1:])
    v_mid = 0.5 * (cone["v_edges"][:-1] + cone["v_edges"][1:])
    total = flux.sum()
    cen_u = float((flux.sum(axis=0) * u_mid).sum() / total)
    cen_v = float((flux.sum(axis=1) * v_mid).sum() / total)
    uu, vv = np.meshgrid(u_mid, v_mid)
    rms = float(np.sqrt((((uu - cen_u) ** 2 + (vv - cen_v) ** 2) * flux).sum() / total))
    return (cen_u, cen_v), rms


def _mc_pooled_xy(pointing_error_mrad: float, n_instants: int, n_rays: int, base_seed: int, trace_args: tuple):
    """Every landed ray's receiver ``(u, v)`` mm, pooled across
    ``n_instants`` INDEPENDENT traces (independent seeds -- independent
    quasi-static pointing draws), the Monte Carlo ensemble-average
    counterpart to a cone kernel's broadened, deterministic spot.
    ``trace_args`` is the fixed geometry (everything ``trace_heliostat``
    takes before ``n_rays``/``rng``).

    Pinned to the super-Gaussian sampler explicitly: this helper's only
    caller (``test_cone_matches_mc_ensemble_but_not_a_single_mc_instant``)
    compares against a cone kernel built with ``sunshape_kernel("super_gauss")``
    -- both sides of that comparison must share one sunshape, or the RMS
    figures are apples to oranges regardless of which shape is the app's
    live default."""
    xs, ys = [], []
    for i in range(n_instants):
        rng = np.random.default_rng((base_seed, i))
        out = trace_heliostat(
            *trace_args, n_rays, rng, pointing_error_mrad=pointing_error_mrad,
            sampler=SuperGaussSampler(),
        )
        xs.append(out["xy"][0])
        ys.append(out["xy"][1])
    return np.concatenate(xs), np.concatenate(ys)


def _rms_about_own_centroid(x: np.ndarray, y: np.ndarray) -> float:
    cx, cy = float(np.mean(x)), float(np.mean(y))
    return float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))


# ---------------------------------------------------------------------------
# 1. zero pointing error is bit-identical, both backends


def test_zero_pointing_error_is_bit_identical_mc_legacy_rect():
    """Omitting ``pointing_error_mrad`` and passing it explicitly at 0.0
    must draw exactly the same random numbers and land exactly the same
    rays -- the ``if pointing_error_mrad:`` gate consumes nothing from
    ``rng`` at zero, so an old caller (or an old project, §F: "nothing
    changes for existing projects") reproduces its trace bit for bit."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    args = (0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
            secondary, receiver, 5000)

    baseline = trace_heliostat(*args, np.random.default_rng(2026), return_paths=True)
    explicit_zero = trace_heliostat(
        *args, np.random.default_rng(2026), return_paths=True, pointing_error_mrad=0.0
    )

    assert np.array_equal(baseline["xy"], explicit_zero["xy"])
    assert np.array_equal(baseline["paths"], explicit_zero["paths"])
    assert baseline["counters"] == explicit_zero["counters"]


def test_zero_pointing_error_is_bit_identical_mc_faceted_design():
    """Same guarantee on the faceted-design code path (§F applies to every
    design type, not just the legacy rectangle) -- a single flat facet at
    the pivot, exercising the ``else`` branch of ``trace_heliostat``'s
    mirror-hit block."""
    from heliostat.geometry.aperture import Rect
    from heliostat.geometry.design import Facet, Flat, HeliostatDesign

    design = HeliostatDesign([Facet(region=Rect(5000.0, 3000.0), surface=Flat())])
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    args = (0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
            secondary, receiver, 5000)

    baseline = trace_heliostat(*args, np.random.default_rng(77), design=design)
    explicit_zero = trace_heliostat(
        *args, np.random.default_rng(77), design=design, pointing_error_mrad=0.0
    )
    assert np.array_equal(baseline["xy"], explicit_zero["xy"])
    assert baseline["counters"] == explicit_zero["counters"]


def test_zero_pointing_error_is_bit_identical_cone_kernel():
    """The cone kernel omitting ``pointing_error_mrad`` and passing it
    explicitly at 0.0 must be the identical kernel -- no broadening
    convolution applied either way (``sunshape_kernel``'s ``broadening > 0``
    gate)."""
    base = sunshape_kernel("super_gauss")
    explicit_zero = sunshape_kernel("super_gauss", pointing_error_mrad=0.0)
    assert base.rms_radius_rad() == explicit_zero.rms_radius_rad()


def test_zero_pointing_error_is_bit_identical_cone_trace():
    """End to end: a cone trace with ``pointing_error_mrad=0`` (the kernel
    construction site ``heliostat.web.app._trace_core`` uses) matches one
    that never mentions pointing error at all."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    args = (0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL, secondary, receiver)

    baseline = trace_heliostat_cone(*args, sunshape_kernel("super_gauss"))
    explicit_zero = trace_heliostat_cone(*args, sunshape_kernel("super_gauss", pointing_error_mrad=0.0))
    assert baseline["power_w"] == explicit_zero["power_w"]
    assert np.array_equal(baseline["flux"], explicit_zero["flux"])


# ---------------------------------------------------------------------------
# 2. MC statistical pin: realised reflected-beam RMS equals the quoted mrad


POINTING_MRAD = 3.0
N_INSTANTS = 300


def test_mc_realised_beam_rms_matches_quoted_mrad():
    """Over many independent timesteps, the per-instant spot-centroid
    deviation's tangential/sagittal RMS (each isolating the ONE shared
    pointing draw that instant made, per ``_mean_tang_sag_deviation``) must
    equal the quoted ``POINTING_MRAD`` per axis -- the mirror-tilt-vs-beam
    factor-of-two bookkeeping (draw sigma = mrad/2, reflection doubles it
    straight back) pinned end to end, measured from receiver geometry, not
    from the internal draw."""
    receiver = _near_normal_receiver()
    tangs, sags = [], []
    for i in range(N_INSTANTS):
        rng = np.random.default_rng((13, i))
        tang, sag, n_landed = _mean_tang_sag_deviation(
            POINTING_MRAD, rng, NORMAL_ROT_AZ, NORMAL_ROT_EL, NORMAL_SUN_AZ, NORMAL_SUN_EL, receiver
        )
        assert n_landed > N_RAYS * 0.2, f"instant {i}: unexpectedly low landing fraction"
        tangs.append(tang)
        sags.append(sag)
    tangs, sags = np.array(tangs), np.array(sags)

    rms_tang_mrad = float(np.sqrt(np.mean(tangs**2))) * 1e3
    rms_sag_mrad = float(np.sqrt(np.mean(sags**2))) * 1e3

    # se(rms)/rms ~ 1/sqrt(2 * N_INSTANTS) ~ 4.1% at N_INSTANTS=300; a 15%
    # band is a comfortable multiple of that without hiding a bookkeeping
    # error the size of a missing/doubled factor of two (50%/100% off).
    assert rms_tang_mrad == pytest.approx(POINTING_MRAD, rel=0.15), (
        f"tangential beam RMS {rms_tang_mrad:.3f} mrad vs quoted {POINTING_MRAD} mrad"
    )
    assert rms_sag_mrad == pytest.approx(POINTING_MRAD, rel=0.15), (
        f"sagittal beam RMS {rms_sag_mrad:.3f} mrad vs quoted {POINTING_MRAD} mrad"
    )

    # Isotropic (both axes drawn with the same per-axis sigma): the ratio
    # should sit near 1.0, not skewed the way a wrong-axis bug would show.
    ratio = rms_tang_mrad / rms_sag_mrad
    assert ratio == pytest.approx(1.0, rel=0.25), f"tang/sag ratio {ratio:.3f} not isotropic"


def test_specularity_still_isotropic_with_pointing_error_present():
    """A cheap non-regression check that pointing error's single shared
    tangent-plane addition (see ``trace_heliostat``'s ``pointing_delta``)
    does not disturb ``specularity_mrad``'s own oblique-incidence isotropy
    fix (``test_mc_physics_fixes.test_specularity_isotropic_about_reflected_ray``)
    -- both perturbations apply to independent things (a per-ray scatter of
    ``d``, a per-call shift of the pre-reflection normal) so they should
    simply add, not interact."""
    rng = np.random.default_rng(4242)
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    out = trace_heliostat(
        0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
        secondary, receiver, 200_000, rng,
        specularity_mrad=10.0, pointing_error_mrad=POINTING_MRAD, return_paths=True,
    )
    assert out["paths"].shape[2] > 1000, "unexpectedly low landing fraction"


# ---------------------------------------------------------------------------
# 3. cone modes: kernel variance bookkeeping, and cone-vs-MC-ensemble parity


def test_cone_kernel_adds_variance_with_no_doubling():
    """Analytic counterpart to ``test_cone_convergence.TestSlopeErrorBroadening``:
    ``pointing_error_mrad`` broadens the kernel by an isotropic 2-D Gaussian
    of sigma = ``pointing_error_mrad`` mrad exactly (NO factor of two,
    unlike ``slope_error_mrad``'s sigma = 2 * slope_error_mrad) -- convolving
    an isotropic 2-D Gaussian adds its variance (``E[theta^2] = 2*sigma^2``)
    to the kernel's own mean-square radius."""
    base = sunshape_kernel("super_gauss")
    broadened = sunshape_kernel("super_gauss", pointing_error_mrad=POINTING_MRAD)
    sigma_broaden = POINTING_MRAD * 1.0e-3
    expected_rms2 = base.rms_radius_rad() ** 2 + 2.0 * sigma_broaden**2
    assert broadened.rms_radius_rad() ** 2 == pytest.approx(expected_rms2, rel=0.02)


def test_cone_kernel_would_be_wrong_if_doubled():
    """Negative control for the test above: doubling ``pointing_error_mrad``
    the way ``slope_error_mrad`` is doubled would broaden the kernel FOUR
    TIMES as much in variance (a factor of 2 in sigma is a factor of 4 in
    sigma^2) -- confirms the two conventions are not accidentally
    equivalent at this magnitude, i.e. this test would actually catch a
    backwards-doubling bug."""
    base = sunshape_kernel("super_gauss")
    broadened = sunshape_kernel("super_gauss", pointing_error_mrad=POINTING_MRAD)
    wrongly_doubled_sigma = 2.0 * POINTING_MRAD * 1.0e-3
    wrongly_doubled_rms2 = base.rms_radius_rad() ** 2 + 2.0 * wrongly_doubled_sigma**2
    assert broadened.rms_radius_rad() ** 2 != pytest.approx(wrongly_doubled_rms2, rel=0.02)


N_ENSEMBLE_INSTANTS = 250
N_ENSEMBLE_RAYS = 4000
# A properly FOCUSED heliostat for this test, unlike the flat (c3=c4=c5=0)
# SPEC_* rectangle above: a flat mirror at 45-deg incidence images the
# source disk almost unfocused (a multi-metre blob, ~1.6 m rms measured),
# which swamps a few-mrad pointing shift and makes "shift vs spread"
# indistinguishable. ``test_mc_parity``'s own prime_focus fixture
# (heliostat 574, mid-morning -- the same one ``test_cone_convergence.py``
# uses for its spot-metric tests) is a real solved twisting figure with a
# compact ~500 mm rms spot, the geometry this comparison actually needs.
_D, _COUNTERS, _SUMMARY = _load_fixture("prime_focus")
_ROW = _SUMMARY.loc[(574, "20260321_0939")]
_SECONDARY, _RECEIVER = _geometry_for("prime_focus")
_FOCUSED_TRACE_ARGS = (
    _ROW.x_mm, _ROW.y_mm, _ROW.rot_az_deg, _ROW.rot_el_deg, _ROW.c3, _ROW.c4, _ROW.c5,
    _ROW.solar_az_deg, _ROW.solar_el_deg, _SECONDARY, _RECEIVER,
)
POINTING_MRAD_ENSEMBLE = 6.0


def test_cone_matches_mc_ensemble_but_not_a_single_mc_instant():
    """The spec's own asymmetry (§F), pinned as one test, on a properly
    focused heliostat:

    * Cone's kernel, broadened by ``pointing_error_mrad``, models the
      LONG-RUN average of many quasi-static instants -- so it should match
      the Monte Carlo ENSEMBLE spot RMS (many independent traces, POOLED),
      not any one of them.
    * A SINGLE Monte Carlo instant is a systematic SHIFT (one shared draw
      for the whole trace), not a spread -- its own spot RMS, measured about
      ITS OWN centroid, should stay close to the unbroadened baseline, not
      the broadened one.
    """
    kernel_base = sunshape_kernel("super_gauss")
    kernel_broadened = sunshape_kernel("super_gauss", pointing_error_mrad=POINTING_MRAD_ENSEMBLE)
    _, rms_cone_base = _cone_centroid_rms_mm(trace_heliostat_cone(*_FOCUSED_TRACE_ARGS, kernel_base))
    _, rms_cone_broadened = _cone_centroid_rms_mm(
        trace_heliostat_cone(*_FOCUSED_TRACE_ARGS, kernel_broadened)
    )
    assert rms_cone_broadened > rms_cone_base, "cone kernel did not broaden at all"

    # MC ensemble: many independent instants, pooled -- the mixture of many
    # randomly-shifted copies of the base spot recovers the same
    # added-variance broadening a convolution does.
    x, y = _mc_pooled_xy(
        POINTING_MRAD_ENSEMBLE, N_ENSEMBLE_INSTANTS, N_ENSEMBLE_RAYS, base_seed=777,
        trace_args=_FOCUSED_TRACE_ARGS,
    )
    rms_mc_ensemble = _rms_about_own_centroid(x, y)

    # 8% band: pooled N is large (N_ENSEMBLE_INSTANTS * N_ENSEMBLE_RAYS *
    # landing-fraction, several hundred thousand rays), so shot noise alone
    # is well under 1%; the rest of the band covers the cone/MC parity-style
    # tolerance already used elsewhere (tests/test_cone_vs_mc.py's own 1.5%
    # sanity band is against a single fixture case with far less pooled N
    # than this ensemble; the pointing-error-specific broadening added here
    # is on top of that baseline agreement, so a slightly wider band).
    assert rms_mc_ensemble == pytest.approx(rms_cone_broadened, rel=0.08), (
        f"MC ensemble rms {rms_mc_ensemble:.2f} mm vs cone broadened rms {rms_cone_broadened:.2f} mm"
    )

    # A single instant: one shared draw for the whole trace, so its OWN
    # spread (about its own, possibly shifted, centroid) should look like
    # the UNBROADENED baseline, not the ensemble figure above -- documenting
    # that a single MC instant legitimately differs from the cone/ensemble
    # picture, per the spec's own design.
    rng = np.random.default_rng((888, 0))
    # Same sampler pin as `_mc_pooled_xy` above: this comparison is against
    # `kernel_base`/`kernel_broadened`, both built with
    # sunshape_kernel("super_gauss") -- the single MC instant must match.
    single = trace_heliostat(
        *_FOCUSED_TRACE_ARGS, N_ENSEMBLE_RAYS, rng, pointing_error_mrad=POINTING_MRAD_ENSEMBLE,
        sampler=SuperGaussSampler(),
    )
    rms_single_instant = _rms_about_own_centroid(single["xy"][0], single["xy"][1])
    assert rms_single_instant == pytest.approx(rms_cone_base, rel=0.15), (
        f"single-instant rms {rms_single_instant:.2f} mm should track the UNBROADENED "
        f"baseline {rms_cone_base:.2f} mm (a shift, not a spread), not the ensemble "
        f"figure {rms_mc_ensemble:.2f} mm"
    )
    assert rms_single_instant < rms_mc_ensemble, (
        "a single quasi-static instant should not show the ensemble's full broadening"
    )


# ---------------------------------------------------------------------------
# 4. quasi-static reproducibility: same seed (+ timestep) -> same offset;
# a different one -> a different offset


def test_same_seed_same_timestep_gives_same_offset():
    """Two traces built from the identical seed (standing in for "the same
    heliostat at the same timestep") must draw the identical pointing
    offset and therefore land bit-identical rays -- not merely close."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    args = (0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
            secondary, receiver, 2000)

    a = trace_heliostat(*args, np.random.default_rng((5, 3)), pointing_error_mrad=POINTING_MRAD)
    b = trace_heliostat(*args, np.random.default_rng((5, 3)), pointing_error_mrad=POINTING_MRAD)
    assert np.array_equal(a["xy"], b["xy"])


def test_different_timestep_gives_a_different_offset():
    """Two traces seeded as "the same heliostat at two different timesteps"
    (only the timestep component of the seed differs) must draw genuinely
    different pointing offsets -- measured as a materially different spot
    centroid, not just non-identical arrays (which any RNG draw at all
    would already guarantee, offset or not)."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()

    def centroid(step_key):
        rng = np.random.default_rng((5, step_key))
        out = trace_heliostat(
            0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
            secondary, receiver, 4000, rng, pointing_error_mrad=POINTING_MRAD,
        )
        return float(np.mean(out["xy"][0])), float(np.mean(out["xy"][1]))

    centroids = [centroid(step) for step in range(6)]
    # Spread across timesteps' centroids should be a sizeable fraction of
    # the pointing-induced displacement itself (mrad * slant range ~ 3e-3 *
    # 3.5e4 mm ~ 100 mm) -- not the sub-mm scatter same-seed noise would
    # give. Using the spread of six independent draws as its own witness
    # avoids hard-coding an absolute mm figure tied to this fixture's exact
    # geometry.
    xs = np.array([c[0] for c in centroids])
    spread_mm = float(np.std(xs))
    assert spread_mm > 20.0, (
        f"centroid x spread across 6 distinct timesteps is only {spread_mm:.2f} mm -- "
        "timesteps do not look independently redrawn"
    )


def _centroid(out: dict) -> tuple[float, float]:
    return float(np.mean(out["xy"][0])), float(np.mean(out["xy"][1]))


def test_pointing_rng_decouples_the_offset_from_the_ray_sampling_seed():
    """The mechanism ``heliostat.web.app._trace_instant_metrics`` relies on
    for a day/year sweep: passing an explicit ``pointing_rng`` draws the
    offset from THAT generator instead of ``rng``, so the SHIFT it adds to
    the spot centroid -- centroid(pointing on) minus centroid(pointing off,
    SAME ``rng``) -- must come out the same whether ``rng`` (ray sampling)
    is seed 101 or seed 202, as long as ``pointing_rng`` is the same seed
    both times. Comparing the SHIFT rather than the raw centroids sidesteps
    a large, unrelated confound: which exact rays land at all is itself
    ``rng``-dependent (a ~40 mm centroid swing between seed 101 and 202 even
    at zero pointing error, this fixture's own binomial landing-fraction
    noise, not a pointing-error signal), so raw centroids from different
    ``rng`` seeds are not directly comparable -- their SHIFT relative to
    each seed's own zero-pointing baseline is."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    args = (0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
            secondary, receiver, 6000)

    def shift(rng_seed: int, pointing_seed: int) -> tuple[float, float]:
        on = trace_heliostat(
            *args, np.random.default_rng(rng_seed), pointing_error_mrad=POINTING_MRAD,
            pointing_rng=np.random.default_rng(pointing_seed),
        )
        off = trace_heliostat(*args, np.random.default_rng(rng_seed), pointing_error_mrad=0.0)
        cx_on, cy_on = _centroid(on)
        cx_off, cy_off = _centroid(off)
        return cx_on - cx_off, cy_on - cy_off

    shift_a = shift(rng_seed=101, pointing_seed=999)
    shift_b = shift(rng_seed=202, pointing_seed=999)
    # Same pointing_rng seed, different rng seed -> the same shift (to
    # within ~0.1 mm: not bit-identical, since a different rng seed lands a
    # slightly different SET of rays inside the same fixed window, but the
    # shift itself is a rigid addition to every one of them).
    assert shift_a[0] == pytest.approx(shift_b[0], abs=1.0), (shift_a, shift_b)
    assert shift_a[1] == pytest.approx(shift_b[1], abs=1.0), (shift_a, shift_b)

    # A DIFFERENT pointing_rng seed (same rng as the first) must give a
    # materially different shift -- pointing_rng, not rng, controls it.
    shift_c = shift(rng_seed=101, pointing_seed=555)
    assert abs(shift_c[0] - shift_a[0]) > 20.0 or abs(shift_c[1] - shift_a[1]) > 20.0, (
        f"a different pointing_rng seed barely moved the shift: {shift_c} vs {shift_a}"
    )
