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

import numpy as np

from ..geometry.aperture import Rect
from ..geometry.design import HeliostatDesign
from ..geometry.receiver import FlatWindowReceiver, Receiver
from ..geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    Secondary,
    _axicon_tip_geometry,
)
from ..trace.mc import (
    MIRROR_HALF_X_MM,
    MIRROR_HALF_Y_MM,
    _mirror_frame,
    _sun_vector,
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

# Side-trace budget for the cone backends (a few milliseconds) and its
# fixed seed -- the scene must be identical for two identical requests.
SIDE_TRACE_RAYS = 4000
SIDE_TRACE_SEED = 20260818

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


def _receiver_dict(receiver: Receiver) -> dict | None:
    """The receiver window as the client needs it: a flat quad and its facing.

    Only :class:`~heliostat.geometry.receiver.FlatWindowReceiver` is
    described here -- the only receiver the web app builds. A curved
    receiver would need its own mesh, and returning ``None`` keeps the
    client from drawing a plausible-looking wrong rectangle.
    """
    if not isinstance(receiver, FlatWindowReceiver):
        return None
    return {
        "z_mm": _round(receiver.z_mm),
        "half_u_mm": _round(receiver.half_u_mm),
        "half_v_mm": _round(receiver.half_v_mm),
        "facing": receiver.facing,
    }


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

    :returns: JSON-safe dict with ``heliostat`` (facet outline polygons),
        ``secondary`` (``None`` or ``{kind, profile}``), ``receiver``,
        ``sun`` (unit vector toward the sun), ``rays`` and ``rays_source``.
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
        else {"kind": profile[0], "profile": [[_round(r), _round(z)] for r, z in rings]},
        "receiver": _receiver_dict(receiver),
        "sun": [_round_unit(c) for c in sun],
        "rays": _rays_payload(paths, max_rays),
        "rays_source": rays_source,
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
) -> dict:
    """Describe one traced *field* instant for the browser's 3-D view.

    ``heliostats`` is one dict per traced heliostat, carrying exactly what
    the trace itself used: ``id``, ``x_mm``, ``y_mm``, ``rot_az_deg``,
    ``rot_el_deg``, ``c3``/``c4``/``c5``, ``design`` (or ``None`` for the
    legacy rectangle) and ``eta`` (the occlusion efficiency actually applied
    to that heliostat's contribution).

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
    """
    outline = decimate_outline(np.asarray(outline_local_mm, dtype=float), max_vertices)

    polygons = []
    table = []
    for h in heliostats:
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

    sources = _field_ray_sources(len(heliostats), n_sources)
    bundles = []
    for i in sources:
        h = heliostats[i]
        sample = trace_heliostat(
            h["x_mm"],
            h["y_mm"],
            h["rot_az_deg"],
            h["rot_el_deg"],
            h["c3"],
            h["c4"],
            h["c5"],
            solar_az_deg,
            solar_el_deg,
            secondary,
            receiver,
            n_sample_rays,
            # Per-heliostat stream, deterministically derived from the fixed
            # scene seed: one seed for all of them would draw the identical
            # sample pattern on every mirror, which reads as a repeated
            # stencil rather than as a field.
            np.random.default_rng(np.random.SeedSequence((SIDE_TRACE_SEED, i))),
            source_disk_radius_mm="auto",
            return_paths=True,
            design=h["design"],
        )
        bundles.append(sample["paths"])

    per_source = max(1, max_rays // max(1, len(bundles)))
    paths = (
        np.concatenate([_subsample_paths(b, per_source) for b in bundles], axis=2)
        if bundles
        else np.zeros((4, 3, 0))
    )

    profile = _secondary_profile(secondary)
    rings = None if profile is None else profile[1][np.isfinite(profile[1]).all(axis=1)]
    sun = _sun_vector(solar_az_deg, solar_el_deg)

    return {
        "heliostat": [_poly_payload(p) for p in polygons],
        "field": {
            "heliostats": table,
            "silhouette_vertices": int(outline.shape[0]),
            "decimated": bool(np.asarray(outline_local_mm).shape[0] > outline.shape[0]),
            # How many heliostats the drawn bundle actually came from -- the
            # caption says so rather than letting the picture imply that
            # every mirror in the field is sending rays.
            "ray_sources": len(sources),
        },
        "secondary": None
        if profile is None
        else {"kind": profile[0], "profile": [[_round(r), _round(z)] for r, z in rings]},
        "receiver": _receiver_dict(receiver),
        "sun": [_round_unit(c) for c in sun],
        "rays": _rays_payload(paths, max_rays),
        "rays_source": "mc_sample",
    }
