"""Scene geometry for the web GUI's 3-D view: what the trace actually saw.

The flux map answers "how much light landed where"; this module answers
"what shape did the light travel through". It turns one traced instant into
a small JSON-safe description -- facet outlines, the secondary's radial
profile, the receiver window, the sun direction and a bundle of real ray
paths -- that the browser renders with its own hand-rolled projection.

**Visual fidelity only.** Nothing here feeds a number back into the
physics, and two deliberate simplifications are taken for the picture's
sake:

* facets are drawn *flat* in their canted planes. Sag is millimetres over
  metres of aperture (a 30 m-focal sphere sags 0.5 mm across a 1.6 m
  facet), so it is invisible at any camera distance that shows the tower;
* facet outlines are polygonal approximations of the real
  :class:`~heliostat.geometry.aperture.Region` sketches. Rectangles are
  exact (bbox corners); everything else -- flower petals, general CSG --
  is sampled radially by :func:`radial_outline`, the same bisection recipe
  :meth:`~heliostat.geometry.design.HeliostatDesign.silhouette` uses, at
  facet resolution rather than whole-design resolution.

The rays are *not* an approximation: they are the surviving paths of a
real Monte Carlo trace (``trace_heliostat(..., return_paths=True)``),
subsampled deterministically. When the reported metrics come from the cone
backend -- which has no rays at all, it propagates measured Jacobians --
this module runs its own small side trace purely for the picture and marks
the result ``rays_source="mc_sample"`` so the UI can say so. That side
trace never touches the reported metrics or flux.

Two scenes, one shape
---------------------
:func:`build_scene` describes ONE heliostat, facet by facet.
:func:`build_field_scene` describes a whole traced field, one *silhouette*
polygon per heliostat, and carries a parallel ``field.heliostats`` table
(id, position, occlusion efficiency) so the client can colour mirrors by
how much of each is shaded or blocked. Both emit the same top-level keys
(``heliostat``, ``secondary``, ``receiver``, ``sun``, ``rays``,
``rays_source``) built by the same helpers, so the browser's renderer draws
either without branching on which one it got.

World coordinates throughout, millimetres, in the tracers' own frame: x
east, y north, z up, heliostat pivot at ``z = 0``, tower axis at
``x = y = 0``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..geometry.aperture import Rect
from ..geometry.design import HeliostatDesign
from ..geometry.receiver import (
    ApertureClippedReceiver,
    CylinderReceiver,
    FlatWindowReceiver,
    FrustumReceiver,
    Receiver,
)
from ..geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    Secondary,
    _axicon_tip_geometry,
)
from ..trace.mc import (
    MIRROR_HALF_X_MM,
    MIRROR_HALF_Y_MM,
    SOURCE_DIST_MM,
    _mirror_frame,
    _sun_vector,
    _zernike_sag_and_slopes,
    design_facet_frames,
    trace_heliostat,
)

# Cap on ray polylines sent to the browser: 200 translucent lines already
# read as a beam; more is payload, not information.
MAX_SCENE_RAYS = 200

# The field view's own budgets. 300 rays spread over the whole field reads as
# "the field is aiming at the tower" without turning the canvas into a solid
# wash; FIELD_RAY_SOURCES is how many heliostats contribute them (a
# deterministic stride through the field, so the bundle samples near, middle
# and far mirrors rather than one corner).
MAX_FIELD_SCENE_RAYS = 300
FIELD_RAY_SOURCES = 12

# Vertices per heliostat outline in the field view. The occluder silhouette
# the physics uses is a 72-gon; drawing 600 of those is ~1.2 MB of payload for
# detail no one can see at field zoom, so the drawn outline is decimated to
# every third vertex. Visual only -- the occlusion answer is unaffected.
FIELD_SILHOUETTE_VERTICES = 24

# How many heliostat silhouettes the field scene actually draws, independent
# of heliostat.web.app.MAX_FIELD_HELIOSTATS (the endpoint's own trace cap).
# Deliberately decoupled: MAX_FIELD_HELIOSTATS exists so real reference
# fields (Gemasolar 2,650, Crescent Dunes 10,347, Hami 14,500) can actually be
# traced, and raising it must not silently raise the browser's JSON payload
# by the same factor. Chosen from the measured payload: a worst-case field
# (flower-petal facets, whose silhouette is a sampled 72-gon rather than four
# rectangle corners) costs ~1.23 KB/heliostat once serialised (its
# FIELD_SILHOUETTE_VERTICES-vertex outline plus its field.heliostats table
# row), so 1000 heliostats land at ~1.21 MB -- comfortably under the 1.5 MB
# scene budget with headroom left for rays, the secondary profile and the
# receiver. (1200 already lands at ~1.46 MB, with almost no headroom left.)
# See test_field_scene_payload_at_cap_stays_small.
MAX_FIELD_SCENE_HELIOSTATS = 1000

# Side-trace budget for the cone backends (a few milliseconds) and its
# fixed seed -- the scene must be identical for two identical requests.
SIDE_TRACE_RAYS = 4000
SIDE_TRACE_SEED = 20260818

# Miss detection (docs/ui-spec.md 2.3's amber "warning" tier): how much
# bigger than the real aperture the probe secondary is. Large enough that a
# field needing, say, 2x today's aperture still reports a finite
# ``needed_aperture_radius_mm`` instead of being clipped by the probe
# itself; not so large that float precision at the enlarged radius becomes
# an issue for any geometry this app builds.
MISS_PROBE_ENLARGE = 20.0

# Fallback overshoot length for a dropped corner ray's dashed extension,
# used whenever the secondary has no revolved profile to measure a "top
# height" from -- prime focus's NoSecondary (no surface of revolution at
# all, so the ray missed a curved receiver directly off the mirror) as well
# as the degenerate case of a real secondary with no profile. Reuses the
# scene's own "far enough to read as deliberate" source distance rather
# than inventing a new constant.
_MISS_RAY_FALLBACK_LEN_MM = SOURCE_DIST_MM

# How far past the secondary's own top height a dashed miss ray overshoots
# -- shared by field_corner_rays' aperture-rim misses and
# _miss_rays_payload's receiver misses, so both read as the same "this
# family of rays would have overshot about this far" picture rather than
# two different conventions that happen to look similar.
_MISS_RAY_OVERSHOOT_FACTOR = 1.3

# Outline sampling: azimuths per facet, and the radial profile resolution
# the client revolves the secondary from.
OUTLINE_DIRECTIONS = 48
PROFILE_POINTS = 24

# Coordinates are rounded to a tenth of a millimetre before serialising:
# far below anything visible on screen, and it roughly halves the payload.
_ROUND_MM = 1


# ---------------------------------------------------------------------------
# outline extraction


def radial_outline(
    region,
    n_directions: int = OUTLINE_DIRECTIONS,
    n_scan: int = 256,
    n_bisect: int = 40,
) -> np.ndarray:
    """Polygonal outline of a 2-D :class:`~heliostat.geometry.aperture.Region`.

    A region only knows membership, so its boundary has to be *found*: for
    each of ``n_directions`` evenly spaced azimuths about the region's bbox
    centre, a coarse scan out to the bbox half-diagonal locates the
    outermost sample still inside, then ``n_bisect`` bisection steps refine
    the crossing. Returned points are on the inside of the boundary (the
    bisection keeps the interior bracket), so ``region.contains`` is true
    for every one of them.

    Same assumption -- and same failure mode -- as
    :meth:`~heliostat.geometry.design.HeliostatDesign.silhouette`: the
    region must be star-shaped about its bbox centre, i.e. every ray from
    the centre leaves the material exactly once. Every facet sketch this
    package builds is (rectangles, discs, petals). An azimuth that finds no
    interior point at all is interpolated from its neighbours rather than
    raising, since this is a picture; a region with *no* interior point on
    any ray raises.

    :returns: ``(n_directions, 2)`` vertices in the region's own frame, mm,
        counter-clockwise, open (the first vertex is not repeated).
    """
    u0, u1, v0, v1 = region.bbox()
    cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    r_max = 1.05 * float(np.hypot(max(u1 - cu, cu - u0), max(v1 - cv, cv - v0)))
    if r_max <= 0:
        raise ValueError("radial_outline: region has an empty bounding box")

    ang = 2.0 * np.pi * np.arange(n_directions) / n_directions
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    scan = np.linspace(0.0, r_max, n_scan)
    inside = np.asarray(
        region.contains(cu + scan[None, :] * cos_a[:, None], cv + scan[None, :] * sin_a[:, None]),
        dtype=bool,
    )
    # Outermost inside sample per direction (argmax on the reversed row
    # finds the last True); -1 where the whole ray misses.
    found = inside.any(axis=1)
    last = (n_scan - 1) - np.argmax(inside[:, ::-1], axis=1)
    lo = np.where(found, scan[last], 0.0)
    hi = np.where(found, scan[np.minimum(last + 1, n_scan - 1)], 0.0)
    hi = np.where(last >= n_scan - 1, r_max, hi)

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        hit = np.asarray(region.contains(cu + mid * cos_a, cv + mid * sin_a), dtype=bool)
        lo = np.where(hit, mid, lo)
        hi = np.where(hit, hi, mid)

    radius = np.where(found, lo, np.nan)
    if not found.any():
        raise ValueError(
            "radial_outline: no interior point found on any radial ray -- "
            "this region is not star-shaped about its bbox centre"
        )
    if not found.all():
        radius[~found] = np.interp(ang[~found], ang[found], radius[found], period=2.0 * np.pi)

    return np.column_stack([cu + radius * cos_a, cv + radius * sin_a])


def _facet_outline_local(region) -> np.ndarray:
    """A facet region's outline in its own ``(lu, lv)`` frame, mm.

    Rectangles (the grid builder's facets, and the rect anchor) are exact
    from their bbox; anything else goes through :func:`radial_outline`.
    """
    if isinstance(region, Rect):
        u0, u1, v0, v1 = region.bbox()
        return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=float)
    return radial_outline(region)


def _facet_polygons(
    design: HeliostatDesign | None,
    helio: np.ndarray,
    n: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> list[np.ndarray]:
    """World-frame outline polygons, one per facet, exactly as traced.

    ``design=None`` is the tracer's legacy single-rectangle mirror
    (``MIRROR_HALF_X_MM`` x ``MIRROR_HALF_Y_MM`` about the pivot, in the
    heliostat's own plane). A design's facets are placed through
    :func:`~heliostat.trace.mc.design_facet_frames`, so a canted facet is
    drawn in the same tilted plane the rays met.
    """
    if design is None:
        corners = np.array(
            [
                [-MIRROR_HALF_X_MM, -MIRROR_HALF_Y_MM],
                [MIRROR_HALF_X_MM, -MIRROR_HALF_Y_MM],
                [MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM],
                [-MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM],
            ]
        )
        return [helio[None, :] + corners[:, :1] * u[None, :] + corners[:, 1:] * v[None, :]]

    polys = []
    for facet, _nf, fu, fv, centre in design_facet_frames(design, helio, n, u, v):
        local = _facet_outline_local(facet.region)
        polys.append(centre[None, :] + local[:, :1] * fu[None, :] + local[:, 1:] * fv[None, :])
    return polys


# ---------------------------------------------------------------------------
# secondary + receiver


def _secondary_profile(secondary: Secondary, n_points: int = PROFILE_POINTS):
    """``(kind, profile)`` radial polyline for a surface of revolution.

    ``profile`` is ``[[r_mm, z_mm], ...]`` from the axis out to the
    secondary's aperture radius; the client revolves it about the tower
    axis. Heights come from the surface classes' own equations -- the
    axicon's cone flank plus its blended tip sphere (via
    :func:`~heliostat.geometry.secondary._axicon_tip_geometry`, the same
    helper the tracer intersects against), the Cassegrain's conic sag --
    so the drawn shape cannot drift from the traced one.

    Returns ``None`` for anything with no radial profile: no secondary at
    all (prime focus), or a shape that is not a surface of revolution (the
    pyramid).
    """
    if isinstance(secondary, AxiconSecondary):
        alpha = np.deg2rad(secondary.half_angle_deg)
        h_t, centre_z, radius = _axicon_tip_geometry(
            secondary.tip_model, secondary.tip_radius_mm, alpha, secondary.apex_height_mm
        )
        r = np.linspace(0.0, secondary.aperture_radius_mm, n_points)
        z = secondary.apex_height_mm + r * np.tan(alpha)
        if radius > 0:
            tip = r < h_t
            z[tip] = centre_z - np.sqrt(np.maximum(radius * radius - r[tip] * r[tip], 0.0))
        return "axicon", np.column_stack([r, z])

    if isinstance(secondary, CassegrainSecondary):
        r = np.linspace(0.0, secondary.aperture_radius_mm, n_points)
        big_r, kk = secondary.vertex_radius_mm, secondary.conic
        h2 = r * r
        root = np.sqrt(np.maximum(1.0 - (1.0 + kk) * h2 / (big_r * big_r), 0.0))
        z = secondary.vertex_z_mm + h2 / (big_r * (1.0 + root))
        return "cassegrain", np.column_stack([r, z])

    return None


def _secondary_perturbation_payload(secondary: Secondary) -> dict:
    """Spec §E2 rigid-body misalignment fields, mm/mrad, for the scene's
    ``secondary`` block.

    ``_secondary_profile`` always draws the surface in its own nominal
    (unperturbed) frame -- the physics-facing consumers care about the
    perturbed surface, but this picture is deliberately the "design"
    geometry, posed at the vertex the client already revolves the profile
    about. Carrying the perturbation here rather than baking it into the
    profile's own points lets the 3-D view pose the mesh (translate by
    decenter, rotate by tip/tilt about the vertex) without this module
    duplicating that transform in Python for a picture only. All zero
    (identity pose) for anything with no rigid-body perturbation to report:
    ``NoSecondary``, ``PyramidSecondary``, or an Axicon/Cassegrain left at
    its default.
    """
    return {
        "dx_mm": _round(getattr(secondary, "dx_mm", 0.0)),
        "dy_mm": _round(getattr(secondary, "dy_mm", 0.0)),
        "dz_mm": _round(getattr(secondary, "dz_mm", 0.0)),
        "tip_mrad": _round(getattr(secondary, "tip_mrad", 0.0)),
        "tilt_mrad": _round(getattr(secondary, "tilt_mrad", 0.0)),
    }


def _receiver_dict(receiver: Receiver) -> dict | None:
    """The receiver as the client needs it to draw: shape, dimensions, facing.

    ``kind`` says which shape it is -- ``"flat"``, ``"cylinder"`` or
    ``"frustum"`` -- with that shape's own fields alongside it; a flat
    receiver's dict keeps exactly the keys it always had
    (``z_mm``/``half_u_mm``/``half_v_mm``/``facing``) plus the new
    ``kind``/``center_x_mm``/``center_y_mm``, so old client code reading
    those keys directly still works unchanged. An
    :class:`~heliostat.geometry.receiver.ApertureClippedReceiver` describes
    its ``inner`` surface this same way, with the clipping ``aperture``
    (its own flat window) attached under an ``"aperture"`` key. Any other
    receiver type returns ``None`` rather than a plausible-looking wrong
    shape.
    """
    aperture = None
    target = receiver
    if isinstance(target, ApertureClippedReceiver):
        ap = target.aperture
        aperture = {
            "z_mm": _round(ap.z_mm),
            "half_u_mm": _round(ap.half_u_mm),
            "half_v_mm": _round(ap.half_v_mm),
            "center_x_mm": _round(ap.center_x_mm),
            "center_y_mm": _round(ap.center_y_mm),
        }
        target = target.inner

    if isinstance(target, FlatWindowReceiver):
        out = {
            "kind": "flat",
            "z_mm": _round(target.z_mm),
            "half_u_mm": _round(target.half_u_mm),
            "half_v_mm": _round(target.half_v_mm),
            "facing": target.facing,
            "center_x_mm": _round(target.center_x_mm),
            "center_y_mm": _round(target.center_y_mm),
        }
    elif isinstance(target, CylinderReceiver):
        out = {
            "kind": "cylinder",
            "center_x_mm": _round(target.center_x_mm),
            "center_y_mm": _round(target.center_y_mm),
            "center_z_mm": _round(target.center_z_mm),
            "radius_mm": _round(target.radius_mm),
            "height_mm": _round(target.height_mm),
        }
    elif isinstance(target, FrustumReceiver):
        out = {
            "kind": "frustum",
            "center_x_mm": _round(target.center_x_mm),
            "center_y_mm": _round(target.center_y_mm),
            "z_bot_mm": _round(target.z_bot_mm),
            "r_bot_mm": _round(target.r_bot_mm),
            "z_top_mm": _round(target.z_top_mm),
            "r_top_mm": _round(target.r_top_mm),
        }
    else:
        return None

    if aperture is not None:
        out["aperture"] = aperture
    return out


def _receiver_center_mm(receiver: Receiver) -> np.ndarray:
    """A representative world point ON (or at the axis of) ``receiver``, mm.

    Not exact for a curved shape -- there is no single "the" point on a
    cylinder or frustum -- but good enough to size a missed ray's dashed
    overshoot (:func:`_miss_rays_payload`) so it actually reads as heading
    toward the receiver's own neighbourhood, whatever the field's scale.
    Unwraps an :class:`ApertureClippedReceiver` to its inner surface, same
    as :func:`_receiver_dict`.
    """
    target = receiver.inner if isinstance(receiver, ApertureClippedReceiver) else receiver
    if isinstance(target, FlatWindowReceiver):
        return np.array([target.center_x_mm, target.center_y_mm, target.z_mm])
    if isinstance(target, CylinderReceiver):
        return np.array([target.center_x_mm, target.center_y_mm, target.center_z_mm])
    if isinstance(target, FrustumReceiver):
        return np.array(
            [target.center_x_mm, target.center_y_mm, 0.5 * (target.z_bot_mm + target.z_top_mm)]
        )
    return np.array([0.0, 0.0, 0.0])  # pragma: no cover - defensive, no other Receiver kind exists


# ---------------------------------------------------------------------------
# rays


def _subsample_paths(paths: np.ndarray, max_rays: int = MAX_SCENE_RAYS) -> np.ndarray:
    """Evenly strided subset of ``(4, 3, K)`` ray paths, at most ``max_rays``.

    Striding rather than a random draw: no rng state, so two identical
    requests give byte-identical rays. The traced rays are already in
    emission order, which is random with respect to position, so a stride
    is an unbiased sample of the bundle.
    """
    k = paths.shape[2]
    if k == 0:
        return paths[:, :, :0]
    step = max(1, int(np.ceil(k / max_rays)))
    return paths[:, :, ::step][:, :, :max_rays]


def _rays_payload(paths: np.ndarray, max_rays: int = MAX_SCENE_RAYS) -> list:
    """``(4, 3, K)`` world paths -> JSON polylines, non-finite rays dropped."""
    sel = _subsample_paths(paths, max_rays)
    finite = np.isfinite(sel).all(axis=(0, 1))
    sel = sel[:, :, finite]
    return [
        [[_round(sel[vtx, axis, i]) for axis in range(3)] for vtx in range(4)]
        for i in range(sel.shape[2])
    ]


def _miss_rays_payload(
    miss_paths: np.ndarray | None,
    miss_dirs: np.ndarray | None,
    secondary: Secondary,
    receiver: Receiver,
    max_rays: int = MAX_SCENE_RAYS,
) -> list:
    """:func:`~heliostat.trace.mc.trace_heliostat`'s ``(3, 3, J)`` miss
    triplets (source, mirror hit, secondary exit) plus their ``(3, J)`` exit
    directions -> the same dashed-red polyline shape
    :func:`field_corner_rays`'s aperture-rim misses already draw, extended
    one more vertex: secondary exit + an overshoot along the real exit
    direction.

    The overshoot length is the LARGER of two things, per ray:
    :data:`_MISS_RAY_OVERSHOOT_FACTOR` x the secondary's own top height
    (:func:`field_corner_rays`'s aperture-rim convention, unchanged for
    axicon/Cassegrain, where the secondary sits close enough to the
    receiver that this alone already overshoots past it), and
    :data:`_MISS_RAY_OVERSHOOT_FACTOR` x that ray's own distance to the
    receiver's centre (:func:`_receiver_center_mm`). The second term is
    what actually matters for prime focus: its fallback overshoot is a
    fixed 30 m (no secondary to measure a height from), which comfortably
    covers a near heliostat but leaves a dashed line from one 90 m out
    stopping in open air, nowhere near the receiver it missed -- this ray
    picked the exact case up (Ryker's own repro: a shrunk prime-focus
    frustum's far corner rays trailed off well short of the tower). Taking
    the max rather than replacing the secondary-based term keeps the
    aperture-rim picture exactly as it was.

    ``None`` (no side-trace miss data, e.g. an older caller) or zero misses
    both return ``[]`` rather than erroring -- a scene with nothing to warn
    about is the common case, not an edge one.
    """
    if miss_paths is None or miss_dirs is None or miss_paths.shape[2] == 0:
        return []
    secondary_len = _MISS_RAY_OVERSHOOT_FACTOR * _secondary_top_height_mm(secondary)
    centre = _receiver_center_mm(receiver)
    dist_to_centre = np.linalg.norm(miss_paths[2] - centre[:, None], axis=0)
    ext_len = np.maximum(secondary_len, _MISS_RAY_OVERSHOOT_FACTOR * dist_to_centre)
    overshoot = miss_paths[2] + ext_len[None, :] * miss_dirs
    full = np.concatenate([miss_paths, overshoot[None, :, :]], axis=0)
    return _rays_payload(full, max_rays)


# ---------------------------------------------------------------------------
# assembly


def _round(x) -> float:
    return round(float(x), _ROUND_MM)


def _round_unit(x) -> float:
    """Rounding for direction cosines -- 0.1 mm is nonsense on a unit vector."""
    return round(float(x), 6)


def _poly_payload(poly: np.ndarray) -> list:
    return [[_round(c) for c in vertex] for vertex in poly]


def build_scene(
    design: HeliostatDesign | None,
    helio_x_mm: float,
    helio_y_mm: float,
    rot_az_deg: float,
    rot_el_deg: float,
    c3: float,
    c4: float,
    c5: float,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary,
    receiver: Receiver,
    paths: np.ndarray | None = None,
    miss_paths: np.ndarray | None = None,
    miss_dirs: np.ndarray | None = None,
    max_rays: int = MAX_SCENE_RAYS,
    n_sample_rays: int = SIDE_TRACE_RAYS,
) -> dict:
    """Describe one traced instant for the browser's 3-D view.

    Every argument that names geometry is the same value the trace itself
    was given, so the picture and the numbers cannot disagree: the pointing
    angles, the figure coefficients, the design (or ``None`` for the legacy
    rectangle), the secondary and the receiver.

    ``paths`` is the Monte Carlo backend's ``(4, 3, K)`` path array when the
    reported trace was a Monte Carlo one. Pass ``None`` for the cone
    backends, which carry no rays: a small seeded side trace
    (``n_sample_rays``) is run here purely to draw a bundle, and the result
    is labelled ``rays_source="mc_sample"``. That side trace is a separate
    call with its own rng and its output never reaches the reported
    metrics.

    ``miss_paths``/``miss_dirs`` are :func:`~heliostat.trace.mc.trace_heliostat`'s
    own ``miss_paths``/``miss_dirs`` (source -> mirror hit -> secondary exit,
    and the direction each left the secondary in) for rays that survived the
    mirror -- and the secondary, if any -- but never reached the receiver's
    window. Only meaningful alongside an explicit ``paths``; when ``paths``
    is ``None`` these are ignored and the side trace's own miss arrays are
    used instead, so the picture's hits and misses always come from the same
    sample. docs/ui-spec.md 2.1's "rays that miss the optics ... draw dashed
    red rather than disappearing" previously only covered the aperture-rim
    case (:func:`field_corner_rays`'s ``return_misses``); a curved
    receiver's own finite extent (a shrunk cylinder/frustum height) drops
    rays this same way, and :func:`_miss_rays_payload` draws them through
    the identical dashed-red pathway.

    :returns: JSON-safe dict with ``heliostat`` (facet outline polygons),
        ``secondary`` (``None`` or ``{kind, profile}``), ``receiver``,
        ``sun`` (unit vector toward the sun), ``rays``, ``rays_source`` and
        ``miss_rays``.
    """
    helio = np.array([helio_x_mm, helio_y_mm, 0.0])
    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)

    if paths is None:
        rays_source = "mc_sample"
        sample = trace_heliostat(
            helio_x_mm,
            helio_y_mm,
            rot_az_deg,
            rot_el_deg,
            c3,
            c4,
            c5,
            solar_az_deg,
            solar_el_deg,
            secondary,
            receiver,
            n_sample_rays,
            np.random.default_rng(SIDE_TRACE_SEED),
            source_disk_radius_mm="auto",
            return_paths=True,
            design=design,
        )
        paths = sample["paths"]
        miss_paths = sample["miss_paths"]
        miss_dirs = sample["miss_dirs"]
    else:
        rays_source = "trace"

    profile = _secondary_profile(secondary)
    sun = _sun_vector(solar_az_deg, solar_el_deg)

    # Standard JSON has no NaN or Infinity token and JS's JSON.parse rejects
    # both, so anything non-finite is dropped here rather than shipped: a
    # degenerate facet costs the picture one outline, not the whole response.
    polygons = [p for p in _facet_polygons(design, helio, n, u, v) if np.isfinite(p).all()]
    rings = None if profile is None else profile[1][np.isfinite(profile[1]).all(axis=1)]

    return {
        "heliostat": [_poly_payload(p) for p in polygons],
        "secondary": None
        if profile is None
        else {
            "kind": profile[0],
            "profile": [[_round(r), _round(z)] for r, z in rings],
            **_secondary_perturbation_payload(secondary),
        },
        "receiver": _receiver_dict(receiver),
        "sun": [_round_unit(c) for c in sun],
        "rays": _rays_payload(paths, max_rays),
        "rays_source": rays_source,
        "miss_rays": _miss_rays_payload(miss_paths, miss_dirs, secondary, receiver, max_rays),
    }


# ---------------------------------------------------------------------------
# the field scene


def decimate_outline(outline: np.ndarray, max_vertices: int = FIELD_SILHOUETTE_VERTICES):
    """Thin a closed outline to at most ``max_vertices``, by even stride.

    A stride keeps the vertices that are actually on the real boundary
    (every kept point is one the silhouette trace found) and keeps them in
    order, so the decimated polygon is inscribed in the original rather than
    smoothed off it. Returns the input untouched when it is already short
    enough -- a rectangle stays its four exact corners.
    """
    v = outline.shape[0]
    if v <= max_vertices:
        return outline
    step = int(np.ceil(v / max_vertices))
    return outline[::step][:max_vertices]


def decimate_heliostats(heliostats: list[dict], max_heliostats: int = MAX_FIELD_SCENE_HELIOSTATS) -> list[dict]:
    """Thin a field's heliostat list to at most ``max_heliostats``, by even
    stride -- the same recipe as :func:`decimate_outline`, applied to
    heliostats instead of vertices.

    A stride rather than a truncation to the first ``max_heliostats``: a
    Fermat spiral or radial-staggered layout is ordered outward from the
    tower, so keeping only a prefix would draw the near field and silently
    drop everything past it. An even stride instead keeps near, middle and
    far heliostats in the same proportion the full field has, so a
    decimated scene still reads as the whole field, just less densely drawn.
    Returns the input untouched when it is already short enough.
    """
    n = len(heliostats)
    if n <= max_heliostats:
        return heliostats
    step = int(np.ceil(n / max_heliostats))
    return heliostats[::step][:max_heliostats]


def _field_ray_sources(n: int, n_sources: int) -> list[int]:
    """Indices of the heliostats that contribute drawn rays.

    An even stride through the field in layout order. For a Fermat spiral
    that order runs outward from the tower, so a stride samples near, middle
    and far mirrors instead of one neighbourhood -- and it uses no rng, so
    two identical requests pick the same heliostats.
    """
    if n <= 0:
        return []
    step = max(1, int(np.ceil(n / n_sources)))
    return list(range(0, n, step))[:n_sources]


def build_field_scene(
    heliostats: list[dict],
    outline_local_mm: np.ndarray,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary,
    receiver: Receiver,
    max_rays: int = MAX_FIELD_SCENE_RAYS,
    n_sources: int = FIELD_RAY_SOURCES,
    n_sample_rays: int = SIDE_TRACE_RAYS,
    max_vertices: int = FIELD_SILHOUETTE_VERTICES,
    max_heliostats: int = MAX_FIELD_SCENE_HELIOSTATS,
) -> dict:
    """Describe one traced *field* instant for the browser's 3-D view.

    ``heliostats`` is one dict per traced heliostat, carrying exactly what
    the trace itself used: ``id``, ``x_mm``, ``y_mm``, ``rot_az_deg``,
    ``rot_el_deg``, ``c3``/``c4``/``c5``, ``design`` (or ``None`` for the
    legacy rectangle) and ``eta`` (the occlusion efficiency actually applied
    to that heliostat's contribution). When the field is bigger than
    ``max_heliostats``, it is thinned first by :func:`decimate_heliostats`
    (an even stride, same idea as the outline's own vertex decimation below)
    -- the trace itself still covers every heliostat and ``eta_min``/
    ``eta_median``/``eta_max`` in the endpoint's own response are computed
    from all of them; only *this picture* draws a representative subset, to
    keep the payload bounded independent of how high
    ``heliostat.web.app.MAX_FIELD_HELIOSTATS`` (the trace cap) is set.

    ``outline_local_mm`` is the design's silhouette in the mirror's own
    ``(u, v)`` plane -- ONE polygon for the whole mirror, not one per facet.
    It is passed in rather than derived here because the caller has already
    built it for the occlusion pass, and because it is the same polygon for
    every heliostat in the field: a silhouette depends on the facet regions
    and their offsets, neither of which moves when a heliostat's own slant
    range changes its figure or cant. Recomputing it per heliostat would
    cost 600 radial traces to get 600 identical answers.

    Rays are always a seeded side trace from a stride of heliostats (see
    :func:`_field_ray_sources`), labelled ``rays_source="mc_sample"`` even
    when the reported field trace was Monte Carlo. Keeping one ray path for
    the picture means the drawn bundle is the same whichever backend ran,
    and a field-wide Monte Carlo carries far more paths than a picture can
    use anyway.

    ``miss_rays`` carries the same stride's corner rays that reflected off
    their mirror (and secondary, if any) but never reached the receiver --
    dashed-red polylines from :func:`field_corner_rays`'s ``return_misses``,
    the picture half of the same "rays that miss ... draw dashed red rather
    than disappearing" contract (docs/ui-spec.md 2.1) that
    :func:`build_geometry_scene` already exposes for the no-trace view.
    """
    outline = decimate_outline(np.asarray(outline_local_mm, dtype=float), max_vertices)
    shown = decimate_heliostats(heliostats, max_heliostats)

    polygons = []
    table = []
    for h in shown:
        centre = np.array([h["x_mm"], h["y_mm"], 0.0])
        _n, u, v = _mirror_frame(h["rot_az_deg"], h["rot_el_deg"])
        poly = centre[None, :] + outline[:, :1] * u[None, :] + outline[:, 1:] * v[None, :]
        if not np.isfinite(poly).all():  # pragma: no cover - defensive, as build_scene
            continue
        polygons.append(poly)
        table.append(
            {
                "id": int(h["id"]),
                "x_mm": _round(h["x_mm"]),
                "y_mm": _round(h["y_mm"]),
                "eta": round(float(h["eta"]), 4),
            }
        )

    # Every DRAWN heliostat contributes four corner chief rays, so the
    # picture shows the (possibly decimated) field working, not a stride of
    # a dozen mirrors that reads as "only these were traced". Built from
    # ``shown`` rather than ``heliostats`` so the rays line up with the
    # silhouettes and table rows actually drawn -- the real trace still
    # covers every heliostat regardless of what this picture shows.
    #
    # return_misses=True so a corner ray that reflects off its mirror (and
    # secondary, if any) but never reaches the receiver -- a shrunk
    # cylinder/frustum height drops rays here that used to just vanish --
    # comes back as the same dashed-red polyline build_geometry_scene's
    # aperture-rim warning already draws, through the identical
    # field_corner_rays pathway.
    corner, field_miss_rays = field_corner_rays(
        shown,
        outline_local_mm,
        solar_az_deg,
        solar_el_deg,
        secondary,
        receiver,
        return_misses=True,
    )

    profile = _secondary_profile(secondary)
    rings = None if profile is None else profile[1][np.isfinite(profile[1]).all(axis=1)]
    sun = _sun_vector(solar_az_deg, solar_el_deg)

    return {
        "heliostat": [_poly_payload(p) for p in polygons],
        "field": {
            "heliostats": table,
            "silhouette_vertices": int(outline.shape[0]),
            # True if either axis of decimation fired: fewer heliostats
            # drawn than traced, or fewer vertices per outline than the
            # real silhouette. Either way the picture is a thinned stand-in
            # for the full field, which is what the client's caption warns
            # about.
            "decimated": bool(
                np.asarray(outline_local_mm).shape[0] > outline.shape[0] or len(shown) < len(heliostats)
            ),
            # Every DRAWN heliostat contributes, so there is no further
            # stride to confess to; kept so the caption can say how the
            # bundle was built.
            "ray_sources": len(table),
        },
        "secondary": None
        if profile is None
        else {
            "kind": profile[0],
            "profile": [[_round(r), _round(z)] for r, z in rings],
            **_secondary_perturbation_payload(secondary),
        },
        "receiver": _receiver_dict(receiver),
        "sun": [_round_unit(c) for c in sun],
        "rays": corner,
        "rays_source": "corner_chief",
        "miss_rays": field_miss_rays,
    }


# ---------------------------------------------------------------------------
# geometry-only scene (no trace)


def build_geometry_scene(
    heliostats: list[dict],
    outline_local_mm: np.ndarray | None,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary | None,
    receiver: Receiver | None,
    *,
    include_corner_rays: bool = True,
    max_corner_sources: int = 500,
    sun_below_horizon: bool = False,
    include_miss_rays: bool = False,
) -> dict:
    """Describe a field's placement and orientation for the 3-D view -- no
    trace, no flux, one solve per heliostat and nothing past it.

    ``heliostats`` is one dict per heliostat: ``id``, ``x_mm``, ``y_mm``, and
    either a solved ``rot_az_deg``/``rot_el_deg``/``c3``/``c4``/``c5``/
    ``design`` -- exactly what :func:`build_field_scene` takes, and what
    :func:`field_corner_rays` needs to reflect a real ray off each mirror's
    own figured surface -- or ``None`` in all five when ``sun_below_horizon``
    is set. There is no aiming solve below the horizon (the per-layout
    solves divide by the sun's own elevation), so those heliostats are
    placed with no orientation rather than a fabricated one; see the
    endpoint's docstring in ``heliostat.web.app`` for why that, and not a
    422, is the answer.

    Unlike :func:`build_field_scene`, this builds no per-heliostat polygon.
    ``outline_local_mm`` -- the mirror's own silhouette, ``(u, v)`` mm, the
    same polygon :func:`_field_geometry` already builds once for the whole
    field's occlusion pass -- is returned once, under ``outline_local``, and
    the caller places one shared mesh at each heliostat's position and
    orientation itself ("instancing" in the graphics sense). That is the
    whole reason this endpoint can afford ten times the heliostat count a
    trace can (:data:`~heliostat.web.app.MAX_GEOMETRY_HELIOSTATS`): the
    payload is O(1) polygons plus O(n) transforms, not O(n) polygons.

    Corner rays reuse :func:`field_corner_rays` verbatim -- real chief rays
    through the real secondary and receiver, deterministic, no shading or
    blocking -- from a stride of at most ``max_corner_sources`` heliostats
    when the field is bigger than that (:func:`_field_ray_sources`), so a
    10,000-heliostat field still draws a bounded number of rays. Skipped
    entirely when ``include_corner_rays`` is false, ``outline_local_mm`` is
    ``None``, the field is empty, or the sun is below the horizon.

    ``include_miss_rays`` (off by default, so every other caller's return
    shape is untouched) additionally routes the same strided sources
    through :func:`field_corner_rays`'s ``return_misses`` and exposes the
    dropped-ray polylines under ``miss_rays`` -- docs/ui-spec.md 2.1's
    unconditional "rays that miss the optics ... draw dashed red rather
    than disappearing", for ANY receiver a corner ray fails to reach
    (including a shrunk prime-focus cylinder/frustum, which has no
    secondary at all). This is independent of
    :func:`field_miss_detection`'s 2.3 amber warning tier -- that one only
    exists for a real secondary and answers "does this heliostat need a
    bigger aperture"; the endpoint computes it separately (it needs every
    heliostat's centre, not a corner-ray stride) and stitches it onto this
    same response, but the two must not be gated on each other.

    :returns: JSON-safe dict with ``outline_local``, ``heliostats`` (id,
        position, orientation), ``secondary``, ``receiver``, ``sun``,
        ``sun_below_horizon``, ``rays``, ``rays_source``
        (``"corner_chief"``, always -- there is no other ray source here),
        and ``miss_rays`` only when ``include_miss_rays`` was set.
    """
    outline = None if outline_local_mm is None else np.asarray(outline_local_mm, dtype=float)

    table = [
        {
            "id": int(h["id"]),
            "x_mm": _round(h["x_mm"]),
            "y_mm": _round(h["y_mm"]),
            "rot_az_deg": None if h.get("rot_az_deg") is None else round(float(h["rot_az_deg"]), 4),
            "rot_el_deg": None if h.get("rot_el_deg") is None else round(float(h["rot_el_deg"]), 4),
            # Already rounded to 3 decimals by the endpoint (_solve_field's
            # own slant, mm -> m); carried through verbatim rather than
            # re-rounded here.
            "slant_range_m": h.get("slant_range_m"),
        }
        for h in heliostats
    ]

    rays: list = []
    result: dict = {}
    if include_corner_rays and not sun_below_horizon and outline is not None and heliostats:
        sources = [heliostats[i] for i in _field_ray_sources(len(heliostats), max_corner_sources)]
        # Used to be gated on having a real secondary (axicon/Cassegrain) --
        # but a prime-focus cylinder/frustum has finite extent too, and a
        # corner ray reflecting straight off the mirror can miss it exactly
        # the same way one can miss a too-small axicon aperture. Excluding
        # NoSecondary here just meant a shrunk curved receiver's dropped
        # rays never made it into the picture (docs/ui-spec.md 2.1's dashed-
        # red contract is unconditional); field_corner_rays' return_misses
        # already handles NoSecondary correctly (secondary.redirect passes
        # every ray through untouched), so there is nothing left to gate on.
        want_misses = include_miss_rays and secondary is not None
        if want_misses:
            rays, result["miss_rays"] = field_corner_rays(
                sources, outline, solar_az_deg, solar_el_deg, secondary, receiver, return_misses=True
            )
        else:
            rays = field_corner_rays(sources, outline, solar_az_deg, solar_el_deg, secondary, receiver)
    if include_miss_rays and "miss_rays" not in result:
        result["miss_rays"] = []

    profile = None if secondary is None else _secondary_profile(secondary)
    rings = None if profile is None else profile[1][np.isfinite(profile[1]).all(axis=1)]
    sun = _sun_vector(solar_az_deg, solar_el_deg)

    result.update(
        {
            "outline_local": None if outline is None else _poly_payload(outline),
            "heliostats": table,
            "secondary": None
            if profile is None
            else {
                "kind": profile[0],
                "profile": [[_round(r), _round(z)] for r, z in rings],
                **_secondary_perturbation_payload(secondary),
            },
            "receiver": None if receiver is None else _receiver_dict(receiver),
            "sun": [_round_unit(c) for c in sun],
            "sun_below_horizon": bool(sun_below_horizon),
            "rays": rays,
            "rays_source": "corner_chief",
        }
    )
    return result


# ---------------------------------------------------------------------------
# field corner rays


def _outline_sample_points(outline_local_mm: np.ndarray, n_points: int = 4) -> np.ndarray:
    """``n_points`` spread around a mirror outline, guaranteed on the mirror.

    Vertices of the outline polygon itself, not corners of its bounding box.
    A bounding box corner is only on the mirror for a rectangle: on a flower
    it falls in the gap between two petals, where there is nothing to reflect
    from, and every ray from it is dropped -- which is how this was found.

    Each point is pulled a little toward the outline's centroid so it lands
    inside the material rather than exactly on its boundary, where a facet
    membership test is a coin flip.
    """
    outline = np.asarray(outline_local_mm, dtype=float)
    n_outline = outline.shape[0]
    if n_outline <= n_points:
        return outline.copy()
    # Half-offset indices, so the samples do not align with the outline's own
    # symmetry axes. Evenly spaced ones do: on a 2x2 facet grid the four gaps
    # run along +-u and +-v, and indices 0, N/4, N/2, 3N/4 land in all four of
    # them, producing a mirror that draws no rays at all.
    idx = np.floor((np.arange(n_points) + 0.5) * n_outline / n_points).astype(int) % n_outline
    return outline[idx]


def _normal_at(h: dict, lu: float, lv: float, n, u, v):
    """Outward normal of one heliostat's surface at local ``(lu, lv)``.

    Carries the figure, which is the whole point: a chief ray reflected off
    the flat plane would draw a diverging beam for a mirror that actually
    focuses. For the legacy single-mirror path the figure is the solve's own
    ``c3``/``c4``/``c5``, negated exactly as
    :func:`heliostat.trace.mc.trace_heliostat` negates them for its inherited
    frame convention. For a design, the facet containing the point supplies
    its own surface and cant; a point that lands on no facet (between a
    flower's petals, say) returns ``None`` and is simply not drawn, rather
    than being drawn against a surface that is not there.
    """
    design = h.get("design")
    if design is None:
        _, dsdx, dsdy = _zernike_sag_and_slopes(
            np.array([lu]), np.array([lv]), h["c3"], -h["c4"], -h["c5"]
        )
        nrm = n - u * float(dsdx[0]) - v * float(dsdy[0])
        return nrm / np.linalg.norm(nrm), np.array([lu, lv])

    helio = np.array([h["x_mm"], h["y_mm"], 0.0])
    point = helio + u * lu + v * lv
    for facet, nf, fu, fv, fcentre in design_facet_frames(design, helio, n, u, v):
        rel = point - fcentre
        fu_l, fv_l = float(rel @ fu), float(rel @ fv)
        if not bool(np.atleast_1d(facet.region.contains(np.array([fu_l]), np.array([fv_l])))[0]):
            continue
        _, dsu, dsv = facet.surface.sag_and_slopes(np.array([fu_l]), np.array([fv_l]))
        nrm = nf - fu * float(dsu[0]) - fv * float(dsv[0])
        return nrm / np.linalg.norm(nrm), np.array([lu, lv])
    return None


def _mirror_frames_batch(rot_az_deg: np.ndarray, rot_el_deg: np.ndarray):
    """Vectorised :func:`~heliostat.trace.mc._mirror_frame` over many
    heliostats at once: the identical construction, ``(N,)`` pointing
    angles in rather than a Python loop of N scalar calls. That loop is
    what :func:`field_corner_rays` already pays per corner-ray *source*
    (bounded by ``max_corner_sources``); :func:`field_miss_detection` runs
    over the whole field instead, so it needs the array form to stay cheap
    at 10,000 heliostats.

    :returns: ``(n, u, v)``, each ``(N, 3)``.
    """
    az = np.deg2rad(rot_az_deg)
    el = np.deg2rad(rot_el_deg)
    n = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, n)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return n, u, v


def _secondary_top_height_mm(secondary: Secondary) -> float:
    """The ``z`` a secondary's surface reaches at its own aperture rim.

    Used only to size the dashed-red overshoot on a dropped corner ray
    (:func:`field_corner_rays`'s ``return_misses``) -- reuses
    :func:`_secondary_profile`, the same revolved profile the 3-D view
    draws the secondary from, so the overshoot length tracks whatever
    surface is actually on screen. Falls back to a fixed scene distance
    for a secondary with no such profile: prime focus's ``NoSecondary``
    (no surface of revolution to measure at all -- a dropped ray there
    missed the receiver directly off the mirror) as well as the degenerate
    case of a real secondary with no profile.
    """
    profile = _secondary_profile(secondary)
    if profile is None:
        return _MISS_RAY_FALLBACK_LEN_MM
    top_z = float(profile[1][-1, 1])
    if not np.isfinite(top_z) or top_z <= 0:
        return _MISS_RAY_FALLBACK_LEN_MM
    return top_z


def field_corner_rays(
    heliostats: list[dict],
    outline_local_mm: np.ndarray,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary,
    receiver: Receiver,
    *,
    return_misses: bool = False,
) -> list | tuple[list, list]:
    """One chief ray from each corner of *every* heliostat in the field.

    Deterministic and cheap: four rays per mirror, the sun's centre only (no
    sunshape), and **no shading or blocking** -- a ray is drawn from every
    corner whether or not a neighbour would have intercepted it. That is a
    deliberate choice for the picture. The sampled Monte Carlo bundle it
    replaces drew a dense spray from a handful of mirrors, which read as
    "only these heliostats were traced"; four rays from all of them reads as
    what the trace actually did. The reported power and flux are unaffected
    either way -- nothing here feeds them.

    The optics are real: each ray reflects off the mirror's own figured
    surface at that corner, then goes through the same secondary and
    receiver objects the trace used.

    ``return_misses=True`` (docs/ui-spec.md 2.1's "Rays that miss the
    optics ... draw dashed red rather than disappearing") additionally
    returns every corner ray that did NOT make it to the receiver -- missed
    the secondary's aperture entirely, or reached the secondary but landed
    outside the receiver window -- as a short ``[source, mirror,
    extension]`` polyline. The extension continues the direction the ray
    left the MIRROR in (the same one a surviving ray uses for its first
    leg), never the post-secondary bounce: a secondary-miss never has one,
    and using one convention for both keeps every dashed line's meaning
    identical regardless of which stage dropped it. Its length is fixed per
    call (1.3x :func:`_secondary_top_height_mm`), not per ray, since this
    is a "this family of rays would have overshot about this far" picture,
    not a traced trajectory.

    :returns: the usual hit-ray polylines
        (``[source, mirror, secondary, receiver]``), or ``(hit_rays,
        miss_rays)`` when ``return_misses`` is set.
    """
    s = _sun_vector(solar_az_deg, solar_el_deg)
    corners = _outline_sample_points(np.asarray(outline_local_mm, dtype=float))

    hits, dirs = [], []
    for h in heliostats:
        n, u, v = _mirror_frame(h["rot_az_deg"], h["rot_el_deg"])
        helio = np.array([h["x_mm"], h["y_mm"], 0.0])
        for lu, lv in corners:
            # A sample can still land on backing structure rather than on a
            # facet -- a gap that reaches the outline, the space between two
            # petals. Walk it toward the mirror's centre until it finds
            # material; give up on that corner if nothing does.
            found = None
            for shrink in (0.98, 0.9, 0.75, 0.5, 0.25):
                slu, slv = float(lu) * shrink, float(lv) * shrink
                found = _normal_at(h, slu, slv, n, u, v)
                if found is not None:
                    break
            if found is None:
                continue
            nrm, local = found
            point = helio + u * float(local[0]) + v * float(local[1])
            d_out = -s - 2.0 * float((-s) @ nrm) * nrm
            hits.append(point)
            dirs.append(d_out)

    if not hits:
        return ([], []) if return_misses else []

    hit = np.asarray(hits, dtype=float).T
    d = np.asarray(dirs, dtype=float).T
    src = hit + SOURCE_DIST_MM * s[:, None]

    pre, d_sec, on_sec = secondary.redirect(hit, d, {})
    reached_sub, uv_sub = receiver.intersect(pre, d_sec)

    # Scatter the (already secondary-filtered) receiver-intersect result
    # back onto the full corner-ray index space, so "reached the receiver"
    # and "everything else" (misses) can both be read off one N-length mask.
    reached = np.zeros(hit.shape[1], dtype=bool)
    reached[on_sec] = reached_sub
    hit_ok = on_sec & reached

    # Receiver.intersect's contract: `reached_sub` masks its (3, K) input,
    # but `uv_sub` comes back ALREADY filtered to the hits, in ray order --
    # so `rec` is aligned with pre[:, reached_sub] as built and must not be
    # masked again. Indexing it with reached_sub a second time was a latent
    # bug inherited from the 3a version of this function; it only detonates
    # when some on-secondary ray never crosses the receiver plane at all
    # (e.g. a 30-degree axicon reflecting steep corner rays upward), because
    # a gentler geometry leaves reached_sub all-True and the double indexing
    # a no-op.
    #
    # World xyz of that hit comes from the receiver's own uv_to_world --
    # exact for any receiver kind, not the old bare z_mm lookup that only
    # existed on FlatWindowReceiver and left every corner ray on a cylinder
    # or frustum with a NaN fourth vertex, dropped whole by the `finite`
    # filter below.
    rec = receiver.uv_to_world(uv_sub)
    paths = np.stack([src[:, hit_ok], hit[:, hit_ok], pre[:, reached_sub], rec])
    finite = np.isfinite(paths).all(axis=(0, 1))
    paths = paths[:, :, finite]
    hit_rays = [
        [[_round(paths[vtx, axis, i]) for axis in range(3)] for vtx in range(4)]
        for i in range(paths.shape[2])
    ]

    if not return_misses:
        return hit_rays

    miss_mask = ~hit_ok
    ext_len = _MISS_RAY_OVERSHOOT_FACTOR * _secondary_top_height_mm(secondary)
    miss_paths = np.stack(
        [
            src[:, miss_mask],
            hit[:, miss_mask],
            hit[:, miss_mask] + ext_len * d[:, miss_mask],
        ]
    )
    miss_finite = np.isfinite(miss_paths).all(axis=(0, 1))
    miss_paths = miss_paths[:, :, miss_finite]
    miss_rays = [
        [[_round(miss_paths[vtx, axis, i]) for axis in range(3)] for vtx in range(3)]
        for i in range(miss_paths.shape[2])
    ]
    return hit_rays, miss_rays


def field_miss_detection(
    heliostats: list[dict],
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary | None,
    receiver: Receiver | None = None,
) -> dict | None:
    """The amber "warning" tier of docs/ui-spec.md 2.3: which heliostats'
    chief rays miss the secondary, and how big an aperture would catch the
    whole field.

    "Catch" means COLLECT, not merely intercept -- Ryker's correction on
    the first cut of this function. Intersecting the (extended) secondary
    surface proves nothing by itself: a 30-degree axicon over the
    manuscript field folds the outer heliostats' light *away* from the
    receiver, so their chief rays cross the extended flank at ~17.3 m and
    then never come down to the window at all. Growing the aperture would
    put glass in front of those rays and deliver nothing. So a heliostat
    only counts as an *aperture* miss if its reflected chief ray, after
    the (enlarged-probe) secondary bounce, actually lands inside the
    receiver window -- i.e. a bigger aperture genuinely fixes it. Every
    other failure -- no secondary intersection at all (the near-heliostat
    case), or a bounce that never reaches the window -- is a *total* miss:
    no aperture size helps.

    ``None`` means "no warning to report", the API contract's three exempt
    cases: no secondary at all (``secondary`` is ``None`` or
    :class:`~heliostat.geometry.secondary.NoSecondary` -- prime focus), or
    an empty field. The fourth exemption, sun below the horizon, is the
    caller's business: there is no solved orientation to build a chief ray
    from then, so this function must not be called in that case (every
    heliostat here needs ``rot_az_deg``/``rot_el_deg``, unlike
    :func:`build_geometry_scene`'s table, which tolerates ``None`` for the
    below-horizon placement-only case).

    Each heliostat's chief ray starts at its own mirror CENTRE, not a
    corner -- this answers "does this heliostat work at all", where
    :func:`field_corner_rays` answers "what does the beam bundle look
    like". It reflects off the plain pointing normal
    (:func:`_mirror_frames_batch`, vectorised over every heliostat at
    once): figure is a fraction of a millimetre of sag at the mirror
    centre, invisible to a which-heliostat-misses test.

    Every ray is tested in ONE vectorised
    :meth:`~heliostat.geometry.secondary.Secondary.redirect` call against
    an enlarged copy of the secondary (:data:`MISS_PROBE_ENLARGE`x the real
    aperture radius, everything else identical via
    :func:`dataclasses.replace`) -- the only way this stays cheap at
    :data:`~heliostat.web.app.MAX_GEOMETRY_HELIOSTATS` heliostats. A ray
    landing inside the REAL aperture is necessarily inside the enlarged one
    too (enlarging only relaxes the aperture cutoff; every other surface
    parameter, including where the surface's root-finding picks its nearest
    intersection, is unchanged), so this single enlarged pass is enough to
    classify every heliostat -- a second call against the real secondary
    would answer nothing this one does not already contain.

    :returns: ``{"needed_aperture_radius_mm", "aperture_miss_ids",
        "total_miss_ids"}``. ``needed_aperture_radius_mm`` is the largest
        hit radius among DELIVERABLE rays only (``None`` if none deliver)
        -- the number is a purchase recommendation, so it must not be
        inflated by rays no aperture can save. ``"rays"`` is deliberately
        absent: the dropped corner-ray polylines come from the corner-ray
        machinery's own strided sources (:func:`field_corner_rays`'s
        ``return_misses``), not from testing all 10,000 heliostats'
        centres, so the caller (:func:`build_geometry_scene`) attaches
        them.
    """
    if secondary is None or isinstance(secondary, NoSecondary) or not heliostats:
        return None

    ids = np.array([h["id"] for h in heliostats], dtype=int)
    x = np.array([h["x_mm"] for h in heliostats], dtype=float)
    y = np.array([h["y_mm"] for h in heliostats], dtype=float)
    az = np.array([h["rot_az_deg"] for h in heliostats], dtype=float)
    el = np.array([h["rot_el_deg"] for h in heliostats], dtype=float)

    n, _u, _v = _mirror_frames_batch(az, el)
    s = _sun_vector(solar_az_deg, solar_el_deg)
    d_in = -s
    d_out = d_in[None, :] - 2.0 * (n @ d_in)[:, None] * n  # (N, 3)

    p = np.column_stack([x, y, np.zeros_like(x)]).T  # (3, N)
    d = d_out.T  # (3, N)

    enlarged = dataclasses.replace(
        secondary, aperture_radius_mm=secondary.aperture_radius_mm * MISS_PROBE_ENLARGE
    )
    hit, d2, on_enlarged = enlarged.redirect(p, d, {})

    # Deliverability: does the post-bounce chief ray land inside the
    # receiver window? Receiver.intersect's contract (same trap as in
    # field_corner_rays): `reached` masks its (3, K) input, `uv` comes back
    # already filtered to the hits in ray order -- clip that against the
    # window extent and scatter back onto the K enlarged-hit columns, then
    # onto the full N. No receiver to test against (a caller that only has
    # a secondary) degrades to the old intercept-only rule.
    deliver_sub = np.ones(hit.shape[1], dtype=bool)
    if receiver is not None and hit.shape[1] > 0:
        reached, uv = receiver.intersect(hit, d2)
        (u0, u1), (v0, v1) = receiver.uv_extent()
        in_window = (uv[0] >= u0) & (uv[0] <= u1) & (uv[1] >= v0) & (uv[1] <= v1)
        deliver_sub = np.zeros(hit.shape[1], dtype=bool)
        deliver_sub[reached] = in_window

    deliverable = np.zeros(len(ids), dtype=bool)
    deliverable[on_enlarged] = deliver_sub

    if hit.shape[1] == 0 or not deliver_sub.any():
        needed = None
        aperture_miss_ids: list = []
    else:
        # `hit` is world-frame (redirect() undoes a spec §E2 rigid-body
        # misalignment on its way out); `aperture_radius_mm` is a property
        # of the physical part's own body, so the comparison needs the same
        # local-frame radius `redirect`'s own rim test used internally, not
        # the point's distance from the world z-axis.
        radius = np.hypot(*secondary.to_local_point(hit)[:2])
        needed = float(np.max(radius[deliver_sub]))
        beyond_actual = (radius > secondary.aperture_radius_mm) & deliver_sub
        aperture_miss_ids = sorted(int(i) for i in ids[on_enlarged][beyond_actual])

    total_miss_ids = sorted(int(i) for i in ids[~deliverable])

    return {
        "needed_aperture_radius_mm": needed,
        "aperture_miss_ids": aperture_miss_ids,
        "total_miss_ids": total_miss_ids,
    }
