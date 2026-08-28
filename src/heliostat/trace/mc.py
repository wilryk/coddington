"""Monte Carlo heliostat trace: one heliostat, one instant, many rays.

Traces the full optical chain sequentially — source disk, figured primary
mirror, secondary reflector, receiver — exactly as the physical light does.
A ray that misses any surface's aperture along the way is dropped there and
never considered again; the counter chain records where.

Convention pins that are not free to change
--------------------------------------------
- Super-Gaussian source: ``I(theta) = exp(-(theta^2 / 2 sigma^2)^n)``,
  ``sigma = 0.0024`` rad, ``n = 2``. See :mod:`heliostat.trace.samplers`.
- Figure sag is the ANSI Z80.28 astigmatic/defocus form, radius normalised
  to 1 mm: ``sag = c3*sqrt(6)*2xy + c4*sqrt(3)*(2x^2+2y^2-1) +
  c5*sqrt(6)*(x^2-y^2)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..geometry.heliostat import zernike_sag_and_slopes as _zernike_sag_and_slopes
from ..geometry.receiver import Receiver
from ..geometry.secondary import Secondary
from .samplers import SuperGaussSampler

if TYPE_CHECKING:
    from ..geometry.design import HeliostatDesign
    from ..geometry.errormap import ErrorMap

# Source geometry: a 3500 mm-radius disk, 30 m from each heliostat along the
# sun vector, emitting toward the mirror. Fixed for every run traced so far.
SOURCE_DISK_RADIUS_MM = 3500.0
SOURCE_DIST_MM = 30000.0
SOURCE_POWER_W = 38484.5  # = 1000 W/m^2 * pi * 3.5^2 m^2

# Mirror aperture half-sizes (rect, mm): 5 m wide along u, 3 m tall along v.
MIRROR_HALF_X_MM = 2500.0
MIRROR_HALF_Y_MM = 1500.0

_DEFAULT_SAMPLER: SuperGaussSampler | None = None


def _default_sampler() -> SuperGaussSampler:
    global _DEFAULT_SAMPLER
    if _DEFAULT_SAMPLER is None:
        _DEFAULT_SAMPLER = SuperGaussSampler()
    return _DEFAULT_SAMPLER


def _sun_vector(solar_az_deg: float, solar_el_deg: float) -> np.ndarray:
    """Unit vector from ground toward the sun (azimuth is compass bearing)."""
    az = np.deg2rad(solar_az_deg)
    el = np.deg2rad(solar_el_deg)
    s = np.array(
        [
            np.cos(el) * np.cos(np.pi / 2 - az),
            np.cos(el) * np.sin(np.pi / 2 - az),
            np.sin(el),
        ]
    )
    return s / np.linalg.norm(s)


def _mirror_frame(rot_az_deg: float, rot_el_deg: float):
    """Mirror normal and in-plane basis from the two stored pointing angles.

    Inverts the ``rot_el = arcsin(n_z)``, ``rot_az = atan2(n_y, n_x)`` map
    and rebuilds ``u``/``v`` the same way the pointing solve does, so the
    frame here is the frame the figure coefficients were computed in.
    """
    az = np.deg2rad(rot_az_deg)
    el = np.deg2rad(rot_el_deg)
    n = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return n, u, v


def _perturb_unit(
    vec: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent per-ray Gaussian tilt of a ``(3, M)`` unit-vector bundle
    within the ``(axis1, axis2)`` tangent plane, renormalised.

    Shared by the slope-error perturbation of a surface normal (before
    reflection) and the specularity perturbation of a ray direction (after
    it) -- same small-angle construction, different vector and different
    tangent axes (the heliostat's own ``(u, v)`` for the legacy rectangle, a
    facet's own ``(fu, fv)`` for a design). ``sigma_rad`` is already the
    per-axis standard deviation in radians; any factor-of-two convention is
    the caller's business, not this helper's.
    """
    m = vec.shape[1]
    out = (
        vec
        + axis1[:, None] * rng.normal(0.0, sigma_rad, m)
        + axis2[:, None] * rng.normal(0.0, sigma_rad, m)
    )
    out /= np.linalg.norm(out, axis=0)
    return out


def _perturb_isotropic(
    d: np.ndarray,
    ref: np.ndarray,
    sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent per-ray isotropic Gaussian scatter of a ``(3, M)`` unit
    direction bundle ``d`` about ITSELF, in the plane perpendicular to each
    ray's own direction.

    This is the specularity convention (matching SolTrace): micro-facet
    roughness scatters the REFLECTED ray about its own ideal direction, not
    about the surface normal, so the two axes spanning the perturbation
    must be perpendicular to ``d`` -- not to the (unperturbed) normal the
    slope-error perturbation (:func:`_perturb_unit`) uses. Those normal-
    tangent axes (mirror ``u``/``v``, or a facet's ``fu``/``fv``) are only
    perpendicular to ``d`` at normal incidence; at oblique incidence they
    are skewed away from perpendicular-to-``d`` by the angle of incidence,
    which is exactly the bug this function fixes (a ``cos(theta)``
    under-broadening of the out-of-plane component).

    ``ref`` is a fixed world direction -- the mirror's own ``u`` for the
    legacy rectangle, a facet's ``fu`` for a design -- used only to seed a
    per-ray basis perpendicular to ``d``: projected out of each ray's own
    ``d`` and renormalised, ``axis1`` is guaranteed perpendicular to ``d``
    (a ray that just reflected off this mirror cannot run parallel to
    ``ref``, which lies in the mirror plane while ``d`` points away from
    it). ``axis2 = d x axis1`` completes a right-handed frame. Because
    reflection about the mirror's true, unperturbed normal is an isometry
    of direction space, an isotropic perturbation built this way in the
    outgoing ray's own perpendicular plane is exactly the SolTrace
    convention and exactly what the cone backend's ``sunshape_kernel``
    assumes when it convolves ``specularity_mrad`` in with no doubling
    (see that function's docstring).

    ``sigma_rad`` is the per-axis standard deviation in radians, the same
    convention :func:`_perturb_unit` uses; this draws exactly two ``(M,)``
    Gaussian arrays from ``rng``, same as that function, so the random-
    number consumption pattern is unaffected by this fix -- only which
    geometric axes the draws land on changes.
    """
    dot_rd = np.einsum("i,ij->j", ref, d)
    axis1 = ref[:, None] - dot_rd[None, :] * d
    axis1 /= np.linalg.norm(axis1, axis=0)
    axis2 = np.cross(d, axis1, axis=0)
    m = d.shape[1]
    out = (
        d
        + axis1 * rng.normal(0.0, sigma_rad, m)
        + axis2 * rng.normal(0.0, sigma_rad, m)
    )
    out /= np.linalg.norm(out, axis=0)
    return out


def design_facet_frames(design, helio: np.ndarray, n: np.ndarray, u: np.ndarray, v: np.ndarray):
    """World-frame geometry per facet: ``(facet, normal, fu, fv, centre)``.

    ``fu`` is the heliostat's ``u`` axis projected into the canted facet's
    plane, ``fv`` completes the right-handed frame — for an uncanted facet
    they are exactly ``(u, v)``, matching where the facet's region and
    surface were authored. Facet centres sit on the heliostat plane at
    their 2-D offsets; the cant tilts the facet about its own centre.
    """
    frames = []
    for facet in design.facets:
        if facet.cant_normal is None:
            nf, fu, fv = n, u, v
        else:
            cn = facet.cant_normal
            nf = cn[0] * u + cn[1] * v + cn[2] * n
            nf = nf / np.linalg.norm(nf)
            fu = u - (u @ nf) * nf
            fu = fu / np.linalg.norm(fu)
            fv = np.cross(nf, fu)
        centre = helio + u * facet.offset_mm[0] + v * facet.offset_mm[1]
        frames.append((facet, nf, fu, fv, centre))
    return frames


def trace_heliostat(
    x_mm: float,
    y_mm: float,
    rot_az_deg: float,
    rot_el_deg: float,
    c3: float,
    c4: float,
    c5: float,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary,
    receiver: Receiver,
    n_rays: int,
    rng: np.random.Generator,
    sampler: SuperGaussSampler | None = None,
    source_disk_radius_mm: "float | str" = SOURCE_DISK_RADIUS_MM,
    source_power_w: float | None = SOURCE_POWER_W,
    return_paths: bool = False,
    return_secondary_hits: bool = False,
    design: "HeliostatDesign | None" = None,
    slope_error_mrad: float = 0.0,
    specularity_mrad: float = 0.0,
    error_map: "ErrorMap | None" = None,
    pointing_error_mrad: float = 0.0,
    pointing_rng: "np.random.Generator | None" = None,
    secondary_error_map: "ErrorMap | None" = None,
    secondary_defocus_um: float = 0.0,
    secondary_astig_um: float = 0.0,
    secondary_astig_axis_deg: float = 0.0,
) -> dict:
    """Trace one heliostat at one instant; return receiver hits and loss counts.

    Sequential: source -> mirror -> secondary -> receiver. A ray that misses
    any surface's aperture is dropped there. Returns a dict with ``xy``
    ``(2, K)`` receiver coordinates inside the receiver's extent and the
    counter chain ``emitted / hit_mirror / hit_secondary / tip_rays /
    reached_receiver / in_window`` (an axicon secondary additionally sets
    ``hit_cone``, equal to ``hit_secondary``, for backward-compatible
    naming). With ``return_paths`` the dict also carries ``paths``: ``(4
    vertices, 3, K)`` world-mm polylines source point -> mirror hit ->
    secondary hit -> receiver hit for every surviving ray; plus
    ``miss_paths`` (``3, 3, J``: source -> mirror hit -> secondary exit) and
    ``miss_dirs`` (``3, J``: the direction each of those ``J`` rays left the
    secondary in) for every ray that reflected off the mirror -- and the
    secondary, if any -- but never reached the receiver's window. Picture
    code (``web/scene.py``) extends each ``miss_dirs`` ray a further fixed
    distance to draw a dashed "missed" overshoot; nothing here decides how
    far.

    ``source_disk_radius_mm`` and ``source_power_w`` default to the values
    every stored trace was generated with; passing different ones changes
    the emitting disk and the reported ``watts_per_ray``, nothing else.
    ``source_disk_radius_mm="auto"`` sizes the disk to the mirror it
    serves — 1.15 x the design's bbox half-diagonal (or the legacy
    rectangle's) — and recomputes ``source_power_w`` accordingly; an
    explicit opt-in so no stored-run convention shifts, but the right
    choice for small or large custom designs, where the fixed 3.5 m disk
    wastes rays or clips corners. Regardless of radius, the disk is always
    CENTRED on the design's own bbox centroid (the legacy rectangle's bbox
    is symmetric about the pivot, so this is a no-op there): a custom
    sketch offset from the pivot -- a hand-drawn outline, a flower whose
    petals don't average back to the origin -- would otherwise have part
    of its area fall outside a disk centred on the pivot instead, silently
    losing the rays that should have illuminated it.

    ``design`` switches the mirror model. ``None`` (default) is the
    original single 5 x 3 m rectangle whose figure is the ``c3``/``c4``/
    ``c5`` terms — that code path is untouched and bit-reproducible
    against the golden fixtures. A :class:`HeliostatDesign` replaces the
    mirror with its facet list: each ray takes the nearest positive facet
    intersection (Newton on that facet's own surface, membership by its
    aperture sketch, reflection off its canted local normal). With a
    design, ``c3``/``c4``/``c5`` are ignored — figures live on the
    design's surfaces, in the design's own frame convention with no
    hidden sign flips (the legacy path negates c4/c5 internally for its
    inherited frame; a design equivalent to legacy ``(c3, c4, c5)``
    therefore carries ``ZernikeAstig(c3, -c4, -c5)``).

    ``slope_error_mrad``/``specularity_mrad`` add per-ray optical error on
    top of whichever figure ran, both zero by default (no perturbation, no
    extra cost, bit-identical to before either existed). ``slope_error_mrad``
    perturbs the mirror's local surface NORMAL, independently per tangent
    axis (``u``/``v``, or a facet's ``fu``/``fv``), before reflection -- a
    manufacturing/mounting tilt, which the reflection law doubles into the
    ray's deflection and which correctly picks up a ``cos(theta_incidence)``
    compression in its out-of-plane component (see :func:`_perturb_unit`).
    ``specularity_mrad`` perturbs the REFLECTED ray direction itself,
    independently per axis, after reflection -- a coating micro-roughness
    scatter, isotropic about the reflected ray by convention (matching
    SolTrace), with no doubling and no incidence-angle compression (see
    :func:`_perturb_isotropic`, which builds its two axes perpendicular to
    the actual outgoing ray rather than to the unperturbed normal). Both
    draw from ``rng``, so a run's random-number sequence is unchanged
    whenever both are left at zero.

    ``pointing_error_mrad`` (docs/ui-spec-v0.2.md §F) is the TRACKER's
    aiming inaccuracy -- "the whole mirror points slightly off its
    commanded direction" -- as opposed to ``slope_error_mrad``'s per-ray
    surface roughness: ONE shared 2-axis Gaussian offset is drawn for the
    WHOLE call (this module's docstring already frames a call as "one
    heliostat, one instant") and added to every ray's/facet's mirror
    normal alike, right after ``slope_error_mrad``/``error_map`` and
    before reflection -- quasi-static per instant, and redrawn only when
    the caller reseeds (or passes a fresh ``pointing_rng``) for a new one.

    The resolved spec convention (§F) is that the quoted
    ``pointing_error_mrad`` is the RMS angular deviation of the REFLECTED
    BEAM, not of the mirror tilt that produces it -- no separate
    factor-of-two is applied to the user's number anywhere (unlike
    ``slope_error_mrad``, which IS the mirror-tilt RMS and picks up its
    factor of two implicitly, from the reflection law itself). Reflection
    doubles a normal tilt into the ray's deflection regardless of what
    perturbed the normal (see the ``dot *= 2.0`` step below, shared by
    every perturbation on this normal) -- so to land the REFLECTED beam's
    RMS exactly on the quoted number, the MIRROR tilt this function draws
    must be HALF of it: ``sigma_rad = (pointing_error_mrad / 2) * 1e-3``
    per axis, doubled straight back to ``pointing_error_mrad`` by the same
    reflection law that already doubles ``slope_error_mrad``. Bookkeeping,
    end to end: draw sigma (mrad, per axis) = pointing_error_mrad / 2 ->
    normal tilt (rad, per axis) = sigma * 1e-3 -> reflection doubles a
    normal tilt into ray deflection -> realised beam deflection (rad, per
    axis) = 2 * sigma * 1e-3 = pointing_error_mrad * 1e-3, i.e. exactly the
    quoted number back in radians. The cone backend's
    ``sunshape_kernel(pointing_error_mrad=...)`` folds in the SAME
    ``pointing_error_mrad`` with no doubling of its own, for exactly this
    reason -- see that function's docstring.

    ``pointing_rng``, if given, draws the pointing offset from a SEPARATE
    generator instead of ``rng`` -- for a caller that must redraw the
    offset every timestep while keeping ``rng`` (and therefore ray
    sampling, ``slope_error_mrad``, ``specularity_mrad``) on the SAME seed
    across timesteps, preserving an existing per-heliostat reproducibility
    guarantee unrelated to pointing error (the web app's day/year sweep:
    see ``heliostat.web.app._trace_instant_metrics``). ``None`` (default)
    draws from ``rng`` like every other error term here -- correct and
    sufficient for a single-instant call (a single-heliostat trace, or a
    field trace), where a fresh call already IS a fresh instant, so
    whatever seed the caller used for ``rng`` already answers "reproducible
    from the seed, redrawn for a new instant." Zero ``pointing_error_mrad``
    draws nothing from either generator -- bit-identical to before this
    parameter existed, exactly like ``slope_error_mrad``/``specularity_mrad``
    at zero.

    ``error_map`` (docs/ui-spec-v0.2.md §E) is a measured/FEA
    :class:`~heliostat.geometry.errormap.ErrorMap`, applied ON TOP OF
    whichever analytic figure ran and BEFORE ``slope_error_mrad`` --
    a deterministic per-ray tilt of the local surface normal, bilinear-
    interpolated from the map's precomputed slope grids at each ray's own
    mirror-point (heliostat aperture-frame ``x, y`` -- for a faceted design,
    a facet's local hit converted back to that frame via its
    ``offset_mm``), added along the heliostat's own global ``u``/``v`` axes
    (the frame the map's CSV convention is defined in) rather than a
    canted facet's tilted ``fu``/``fv`` -- consistent with how
    :func:`heliostat.web.app._sag_grid_mm` samples a design's sag in plan
    view. Costs one bilinear lookup per ray regardless of the map's own
    resolution (pre-processed once at import), so Monte Carlo trace time is
    essentially unaffected. ``None`` (default) skips this branch entirely,
    bit-identical to before this parameter existed. There is no cone-mode
    equivalent -- the cone backends never see this parameter, so a map
    changes nothing about their kernels by construction (spec: "Monte Carlo
    only").

    ``secondary_error_map``/``secondary_defocus_um``/``secondary_astig_um``/
    ``secondary_astig_axis_deg`` (docs/ui-spec-v0.2.md §E2) are the
    SECONDARY's own measured error map and parametric warp -- the §E
    machinery above, adapted from the mirror's rectangle to the secondary's
    circular aperture, and from the mirror's global ``u``/``v`` to the
    secondary's own local ``x``/``y``. Forwarded straight into
    ``secondary.redirect()`` as keyword-only arguments; every OTHER caller
    of ``redirect()`` in this codebase (:mod:`heliostat.trace.cone`,
    :mod:`heliostat.web.scene`) calls it without these keywords at all, so
    their defaults (``None``/``0.0``) apply there unconditionally -- that
    default, not a mode check, is what makes this MC-only: cone traces are
    bit-identical whether or not a secondary map/warp is configured, by
    construction, exactly like the primary mirror's own ``error_map``
    above. See :meth:`~heliostat.geometry.secondary.Secondary.redirect`'s
    own docstring and :meth:`~heliostat.geometry.secondary.AxiconSecondary.redirect`/
    :meth:`~heliostat.geometry.secondary.CassegrainSecondary.redirect` for
    where and how the perturbation is actually applied (in the secondary's
    LOCAL unperturbed frame, before the rigid-body transform back to world,
    so it composes correctly with a §E2 decenter/tilt), and
    :func:`~heliostat.geometry.secondary.secondary_warp_sag_mm`/
    :func:`~heliostat.geometry.secondary.secondary_warp_slopes` for the
    parametric-warp closed forms and the composition-with-the-map choice
    (summed, kept analytic rather than baked into one shared grid --
    documented there).
    """
    if sampler is None:
        sampler = _default_sampler()

    if isinstance(source_disk_radius_mm, str):
        if source_disk_radius_mm != "auto":
            raise ValueError("source_disk_radius_mm must be a number or 'auto'")
        half_diag = (
            design.half_diagonal_mm
            if design is not None
            else float(np.hypot(MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM))
        )
        source_disk_radius_mm = 1.15 * half_diag
        source_power_w = 1000.0 * np.pi * (source_disk_radius_mm / 1000.0) ** 2

    # heliostat_shape (../geometry/heliostat.py) computes the figure in a
    # frame whose y and z axes are the mirror's own; the Zernike sag
    # evaluated below is written in a frame with y and z flipped relative to
    # that. Carrying the flip through the ANSI Z3/Z4/Z5 terms negates c4 and
    # c5 while leaving c3 unchanged -- a bookkeeping correction between two
    # frame conventions that agree on everything else, not new physics.
    c4 = -c4
    c5 = -c5

    s = _sun_vector(solar_az_deg, solar_el_deg)
    helio = np.array([x_mm, y_mm, 0.0])
    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)

    # §F pointing error: ONE shared whole-mirror tilt for this entire call,
    # drawn here (before any per-ray draws, so its position in the stream
    # never depends on n_rays or how many rays survive later) and added to
    # every ray's/facet's normal below -- see the docstring above for the
    # /2 bookkeeping and why this consumes `pointing_rng` (falling back to
    # `rng`) rather than a per-ray draw.
    pointing_delta = None
    if pointing_error_mrad:
        prng = pointing_rng if pointing_rng is not None else rng
        sigma_rad = (pointing_error_mrad * 0.5) * 1.0e-3
        off_u, off_v = prng.normal(0.0, sigma_rad, 2)
        pointing_delta = u * off_u + v * off_v

    # Source disk basis: any orthonormal pair perpendicular to the sun
    # vector works -- positions are uniform and the angular law is
    # axisymmetric.
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), s)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)

    # The disk must be centred over the mirror region it actually needs to
    # cover, not over the heliostat's pivot -- for the legacy rectangle
    # those coincide (bbox symmetric about (0, 0)), but a custom design's
    # sketch can sit anywhere in its own (u, v) plane (an outline offset
    # from the pivot, a flower whose petals don't average back to the
    # origin). Recentre on the design's own bbox centroid, carried into
    # world space through the mirror's (u, v) axes and then out along the
    # sun vector to the source plane, so a ray aimed at the disk centre
    # still lands at the mirror region's own centre, not at the pivot.
    if design is not None:
        du0, du1, dv0, dv1 = design.bbox
        centre_u = 0.5 * (du0 + du1)
        centre_v = 0.5 * (dv0 + dv1)
    else:
        centre_u = centre_v = 0.0
    mirror_centre = helio + u * centre_u + v * centre_v
    centre = mirror_centre + SOURCE_DIST_MM * s

    r = source_disk_radius_mm * np.sqrt(rng.random(n_rays))
    phi = 2.0 * np.pi * rng.random(n_rays)
    p = (
        centre[:, None] + e1[:, None] * (r * np.cos(phi)) + e2[:, None] * (r * np.sin(phi))
    )  # (3, N)

    theta = sampler.sample(n_rays, rng)
    psi = 2.0 * np.pi * rng.random(n_rays)
    d = -s[:, None] * np.cos(theta) + (
        e1[:, None] * np.cos(psi) + e2[:, None] * np.sin(psi)
    ) * np.sin(theta)

    counters = {"emitted": n_rays}

    # --- primary mirror -----------------------------------------------
    if design is None:
        dn = d.T @ n  # (N,)
        du, dv = d.T @ u, d.T @ v
        t = ((helio - p.T) @ n) / dn
        # One Newton correction for the sag, then a final evaluation: the
        # figure is millimetres over a 3 m half-aperture, so a second
        # correction is sub-micron -- far below receiver storage quantisation.
        hit = p + d * t
        rel = hit.T - helio
        lx, ly = rel @ u, rel @ v
        sag, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
        t -= (rel @ n - sag) / (dn - dsdx * du - dsdy * dv)
        hit = p + d * t
        rel = hit.T - helio
        lx, ly = rel @ u, rel @ v
        ok = (np.abs(lx) <= MIRROR_HALF_X_MM) & (np.abs(ly) <= MIRROR_HALF_Y_MM) & (t > 0)
        counters["hit_mirror"] = int(ok.sum())
        hit, d, lx, ly = hit[:, ok], d[:, ok], lx[ok], ly[ok]

        _, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
        normal = n[:, None] - u[:, None] * dsdx - v[:, None] * dsdy
        normal /= np.linalg.norm(normal, axis=0)
        if error_map is not None:
            map_dsdx, map_dsdy = error_map.sample_slopes(lx, ly)
            normal = normal - u[:, None] * map_dsdx - v[:, None] * map_dsdy
            normal /= np.linalg.norm(normal, axis=0)
        if slope_error_mrad:
            normal = _perturb_unit(normal, u, v, slope_error_mrad * 1.0e-3, rng)
        if pointing_delta is not None:
            normal = normal + pointing_delta[:, None]
            normal /= np.linalg.norm(normal, axis=0)
        # In-place reflection: d -= 2 (d.n) n, no fresh (3, M) temporaries.
        dot = np.einsum("ij,ij->j", d, normal)
        dot *= 2.0
        d -= dot * normal
        if specularity_mrad:
            d = _perturb_isotropic(d, u, specularity_mrad * 1.0e-3, rng)
    else:
        frames = design_facet_frames(design, helio, n, u, v)
        n_in = d.shape[1]
        t_all = np.full((len(frames), n_in), np.inf)
        lu_all = np.zeros((len(frames), n_in))
        lv_all = np.zeros((len(frames), n_in))
        for k, (facet, nf, fu, fv, centre) in enumerate(frames):
            dn = d.T @ nf
            du, dv = d.T @ fu, d.T @ fv
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((centre - p.T) @ nf) / dn
            hit = p + d * t
            rel = hit.T - centre
            lu, lv = rel @ fu, rel @ fv
            sag, dsu, dsv = facet.surface.sag_and_slopes(lu, lv)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = t - (rel @ nf - sag) / (dn - dsu * du - dsv * dv)
            hit = p + d * t
            rel = hit.T - centre
            lu, lv = rel @ fu, rel @ fv
            valid = facet.region.contains(lu, lv) & (t > 0) & np.isfinite(t)
            t_all[k] = np.where(valid, t, np.inf)
            lu_all[k], lv_all[k] = lu, lv
        # Nearest positive facet intersection wins (overlapping canted
        # facets near a hub genuinely differ in range).
        best = np.argmin(t_all, axis=0)
        ray_idx = np.arange(n_in)
        t_sel = t_all[best, ray_idx]
        ok = np.isfinite(t_sel)
        counters["hit_mirror"] = int(ok.sum())
        hit = p + d * t_sel
        hit, d = hit[:, ok], d[:, ok]
        best = best[ok]
        normal = np.empty_like(d)
        for k, (facet, nf, fu, fv, centre) in enumerate(frames):
            grp = best == k
            if not np.any(grp):
                continue
            grp_lu, grp_lv = lu_all[k][ok][grp], lv_all[k][ok][grp]
            _, dsu, dsv = facet.surface.sag_and_slopes(grp_lu, grp_lv)
            nrm = nf[:, None] - fu[:, None] * dsu - fv[:, None] * dsv
            nrm /= np.linalg.norm(nrm, axis=0)
            if error_map is not None:
                # The map's own frame is the heliostat's aperture-frame
                # plan view (§D convention), not this facet's canted
                # (fu, fv) -- so both the query point and the tangent axes
                # the correction is added along use the heliostat's global
                # (u, v), converting this facet's local hit back to that
                # frame via its offset first.
                full_x = grp_lu + facet.offset_mm[0]
                full_y = grp_lv + facet.offset_mm[1]
                map_dsdx, map_dsdy = error_map.sample_slopes(full_x, full_y)
                nrm = nrm - u[:, None] * map_dsdx - v[:, None] * map_dsdy
                nrm /= np.linalg.norm(nrm, axis=0)
            if slope_error_mrad:
                nrm = _perturb_unit(nrm, fu, fv, slope_error_mrad * 1.0e-3, rng)
            if pointing_delta is not None:
                nrm = nrm + pointing_delta[:, None]
                nrm /= np.linalg.norm(nrm, axis=0)
            normal[:, grp] = nrm
        dot = np.einsum("ij,ij->j", d, normal)
        dot *= 2.0
        d -= dot * normal
        if specularity_mrad:
            sigma = specularity_mrad * 1.0e-3
            for k, (facet, nf, fu, fv, centre) in enumerate(frames):
                grp = best == k
                if not np.any(grp):
                    continue
                d[:, grp] = _perturb_isotropic(d[:, grp], fu, sigma, rng)

    # --- secondary -------------------------------------------------------
    pre, d, on_sec = secondary.redirect(
        hit,
        d,
        counters,
        secondary_error_map=secondary_error_map,
        defocus_um=secondary_defocus_um,
        astig_um=secondary_astig_um,
        astig_axis_deg=secondary_astig_axis_deg,
    )

    # --- receiver ----------------------------------------------------------
    hit_mask, uv = receiver.intersect(pre, d)
    counters["reached_receiver"] = int(hit_mask.sum())
    (u0, u1), (v0, v1) = receiver.uv_extent()
    inside = (uv[0] >= u0) & (uv[0] <= u1) & (uv[1] >= v0) & (uv[1] <= v1)
    counters["in_window"] = int(inside.sum())

    result = {
        "xy": uv[:, inside],
        "counters": counters,
        "source_power_w": source_power_w,
        "watts_per_ray": source_power_w / n_rays if n_rays else 0.0,
    }
    if return_secondary_hits:
        # World (x, y, z) of every ray that struck the secondary, whether or
        # not it reached the receiver window -- for irradiance maps on the
        # secondary itself. The full 3-D point (not just plan x, y) is kept
        # so a spec §E2 rigid-body misalignment can be undone exactly: the
        # world -> local inverse a perturbed secondary_uv() needs (via
        # Secondary.to_local_point) is a rotation that mixes z into x and y,
        # so a hit's z is load-bearing here even though the unperturbed
        # secondary_uv formula itself never reads it.
        result["secondary_xy"] = pre.copy()
    if return_paths:
        # `pre`/`d` are already down to the K = on_sec.sum() rays that
        # struck the secondary (or all of them, for NoSecondary); `hit_mask`
        # is a K-length mask into them, and `uv`/`inside` are already
        # compacted to hit_mask's own survivors (Receiver.intersect's
        # contract -- see the note in web/scene.py's field_corner_rays about
        # not indexing `inside` against the wrong length). Scattering
        # `inside` back onto the K-length space gives one mask, `delivered`,
        # that answers "this on-secondary ray reached the receiver AND
        # landed in its window" -- exactly the old chained
        # `[:, hit_mask][:, inside]` filter, just named so its complement
        # (rays that got through the mirror/secondary but never made it to
        # the receiver) can be captured too.
        src_sec = p[:, ok][:, on_sec]
        mir_sec = hit[:, on_sec]
        delivered = np.zeros(pre.shape[1], dtype=bool)
        delivered[hit_mask] = inside

        src = src_sec[:, delivered]
        mir = mir_sec[:, delivered]
        con = pre[:, delivered]  # == mir when there is no secondary
        rec_uv = uv[:, inside]
        # World xyz of the receiver hit -- exact for any receiver kind via
        # its own uv_to_world (flat, cylinder, frustum, or an aperture-
        # clipped wrapper delegating to whichever of those sits behind it),
        # not a bare z_mm lookup that only exists on the flat case.
        rec = receiver.uv_to_world(rec_uv)
        result["paths"] = np.stack([src, mir, con, rec])

        # Rays that reflected off the mirror (and the secondary, if any)
        # but never reached the receiver -- missed its surface entirely, or
        # landed outside its window -- used to simply vanish: a curved
        # receiver's finite extent (a shrunk cylinder/frustum height) drops
        # rays here that a flat window's infinite-plane test never did.
        # docs/ui-spec.md 2.1 wants these drawn dashed red rather than
        # disappearing; web/scene.py builds that polyline (source, mirror
        # hit, secondary exit, then a picture-only overshoot along `d`) from
        # the raw points and this exit direction returned here, the same way
        # it already does for field_corner_rays' aperture-rim misses.
        miss = ~delivered
        result["miss_paths"] = np.stack([src_sec[:, miss], mir_sec[:, miss], pre[:, miss]])
        result["miss_dirs"] = d[:, miss]
    return result
