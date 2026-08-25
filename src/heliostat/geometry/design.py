"""Heliostat design: the physics layer over aperture sketches.

An :class:`~heliostat.geometry.aperture.Region` only knows 2-D shape --
membership, bbox, area. A :class:`HeliostatDesign` adds what the optical
engine needs on top of that: a *figure* (the mirror's sag and slope in its
own local frame) per facet, where each facet sits in the heliostat's own
``(u, v)`` plane, and how each facet is *canted* (tilted, in 3-D, so its own
reflection converges toward the design's focal point rather than being
parallel to every other facet).

Frames, spelled out because three of them are in play at once:

* the **heliostat frame** ``(u, v)`` -- the whole mirror's own plane, origin
  at the heliostat's pivot, same frame :mod:`heliostat.trace.mc` traces
  rays against;
* a **facet's local frame** ``(lu, lv)`` -- where its :class:`Facet.region`
  is defined and where :meth:`Surface.sag_and_slopes` is evaluated. It sits
  at ``offset_mm`` in the heliostat frame, axes parallel to ``(u, v)`` --
  there is no separate in-plane rotation field on :class:`Facet`. Any facet
  whose footprint should *appear* rotated in the heliostat frame (a flower
  petal pointing outward, say) gets that rotation baked into its own
  ``region``, not into a rotation the Facet applies for it (see
  :func:`_petal_at_angle` for why that is built directly rather than via
  :meth:`~heliostat.geometry.aperture.Region.rotated`);
* **cant** is a purely 3-D, out-of-plane notion -- a unit surface normal in
  the *heliostat* frame that tilts a flat facet's reflection toward the
  focal point without touching its 2-D footprint at all.

:func:`rect_heliostat` is the parity anchor: a single flat/unastigmatic
rectangular facet reproducing today's one-mirror model exactly, so any
engine consuming :class:`HeliostatDesign` must agree with
:mod:`heliostat.trace.mc` on that one case before it is trusted on anything
fancier.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from .aperture import Disc, Polygon, Rect, Region, circular_array, petal, region_from_dict
from .heliostat import zernike_sag_and_slopes as _zernike_sag_and_slopes

# ---------------------------------------------------------------------------
# surfaces


class Surface(ABC):
    """A mirror figure: sag and slope in a facet's own local frame, mm."""

    kind: str = "abstract"

    @abstractmethod
    def sag_and_slopes(self, lu, lv):
        """``(sag, dsdu, dsdv)`` at local coordinates ``(lu, lv)``, mm & mm/mm.

        Broadcasts like :meth:`~heliostat.geometry.aperture.Region.contains`.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-safe description; ``surface_from_dict`` inverts it."""


class Flat(Surface):
    """A plane: zero sag everywhere, zero slope."""

    kind = "flat"

    def sag_and_slopes(self, lu, lv):
        lu, lv = np.broadcast_arrays(np.asarray(lu, dtype=float), np.asarray(lv, dtype=float))
        zero = np.zeros_like(lu)
        return zero, zero.copy(), zero.copy()

    def to_dict(self):
        return {"kind": self.kind}


@dataclass(frozen=True)
class Spherical(Surface):
    """Paraxial spherical cap: ``sag = (lu^2 + lv^2) / (4 f)``.

    ``focal_mm`` is either a number or the literal string ``"slant"``, a
    placeholder meaning "use this facet's own 3-D distance to the design's
    on-axis focal point rather than the nominal focal length" -- the usual
    per-facet correction for a canted facet array, where a facet far from
    the axis sits measurably closer to or farther from the target than the
    design's nominal focal length says. ``"slant"`` is resolved to a number
    by the builder that has the geometry to compute it (:func:`grid_facets`,
    :func:`flower`, when given ``cant_focal_mm``); :meth:`sag_and_slopes`
    raises if it is ever called before that resolution happens.
    """

    focal_mm: float | str

    kind = "spherical"

    def sag_and_slopes(self, lu, lv):
        if isinstance(self.focal_mm, str):
            raise ValueError(
                "Spherical(focal_mm='slant') was never resolved to a number -- "
                "build this facet through grid_facets()/flower() with cant_focal_mm set, "
                "or pass a numeric focal_mm directly"
            )
        f = float(self.focal_mm)
        lu = np.asarray(lu, dtype=float)
        lv = np.asarray(lv, dtype=float)
        sag = (lu * lu + lv * lv) / (4.0 * f)
        dsdu = lu / (2.0 * f)
        dsdv = lv / (2.0 * f)
        return sag, dsdu, dsdv

    def to_dict(self):
        return {"kind": self.kind, "focal_mm": self.focal_mm}


@dataclass(frozen=True)
class ZernikeAstig(Surface):
    """ANSI Z3/Z4/Z5 astigmatic + defocus figure, delegating to the tracer's form.

    Deliberately not reimplemented here: :func:`heliostat.trace.mc._zernike_sag_and_slopes`
    is the pinned convention (see that module's docstring), and this class
    exists only to expose it through the :class:`Surface` interface.
    """

    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0

    kind = "zernike_astig"

    def sag_and_slopes(self, lu, lv):
        return _zernike_sag_and_slopes(lu, lv, self.c3, self.c4, self.c5)

    def to_dict(self):
        return {"kind": self.kind, "c3": self.c3, "c4": self.c4, "c5": self.c5}


_SURFACE_REGISTRY: dict[str, type] = {cls.kind: cls for cls in (Flat, Spherical, ZernikeAstig)}


def surface_from_dict(data: dict) -> Surface:
    """Rebuild a surface from :meth:`Surface.to_dict` output."""
    data = dict(data)
    kind = data.pop("kind")
    cls = _SURFACE_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown surface kind {kind!r}; known: {sorted(_SURFACE_REGISTRY)}")
    return cls(**data)


# ---------------------------------------------------------------------------
# facets and designs


@dataclass(frozen=True)
class Facet:
    """One mirror piece: an aperture sketch, a figure, and where it sits."""

    region: Region
    surface: Surface
    offset_mm: tuple[float, float] = (0.0, 0.0)
    cant_normal: np.ndarray | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        u0, u1, v0, v1 = self.region.bbox()
        ou, ov = self.offset_mm
        return (u0 + ou, u1 + ou, v0 + ov, v1 + ov)

    def to_dict(self) -> dict:
        return {
            "region": self.region.to_dict(),
            "surface": self.surface.to_dict(),
            "offset_mm": [float(self.offset_mm[0]), float(self.offset_mm[1])],
            "cant_normal": None
            if self.cant_normal is None
            else [float(x) for x in self.cant_normal],
        }

    @staticmethod
    def from_dict(data: dict) -> "Facet":
        offset = data.get("offset_mm", (0.0, 0.0))
        cant = data.get("cant_normal")
        return Facet(
            region=region_from_dict(data["region"]),
            surface=surface_from_dict(data["surface"]),
            offset_mm=(float(offset[0]), float(offset[1])),
            cant_normal=None if cant is None else np.asarray(cant, dtype=float),
        )


class HeliostatDesign:
    """A heliostat as a set of facets: what an engine needs beyond one flat rect."""

    def __init__(self, facets: list[Facet]):
        if not facets:
            raise ValueError("a design needs at least one facet")
        self.facets = list(facets)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Union of every facet's bbox, shifted by its offset."""
        boxes = np.array([f.bbox for f in self.facets])
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].max()),
            float(boxes[:, 2].min()),
            float(boxes[:, 3].max()),
        )

    @property
    def area_mm2(self) -> float:
        """Sum of facet region areas (exact where the region supports it)."""
        return float(sum(f.region.area_mm2() for f in self.facets))

    @property
    def half_diagonal_mm(self) -> float:
        """Centre-to-corner distance of :attr:`bbox`, for sizing a source disk."""
        u0, u1, v0, v1 = self.bbox
        return float(np.hypot((u1 - u0) / 2.0, (v1 - v0) / 2.0))

    def to_dict(self) -> dict:
        return {"facets": [f.to_dict() for f in self.facets]}

    @staticmethod
    def from_dict(data: dict) -> "HeliostatDesign":
        return HeliostatDesign([Facet.from_dict(f) for f in data["facets"]])

    # -- silhouette ---------------------------------------------------

    def silhouette(self, n_vertices: int = 72) -> Polygon:
        """The design's outer-perimeter polygon in the ``(u, v)`` plane.

        For shadowing/blocking, an occluder presents its *filled* outline:
        facet gaps and interior holes sit in front of backing structure and
        do not transmit light, so a neighbour is shaded by the whole panel
        silhouette even where the exact sketch has no material. The exact
        facet sketches stay authoritative for what reflects; this silhouette
        is only for what shades or blocks.

        Radial trace about the design's bbox centre: for ``n_vertices``
        evenly spaced azimuths, a coarse scan out to the bbox half-diagonal
        (plus margin) finds the outermost point still inside *some* facet's
        region (``offset_mm``-shifted, ignoring cant -- this is a 2-D
        question), then ~40 bisection steps refine that crossing. This
        assumes the design is star-shaped about its centre: every ray from
        the centre enters the material at most once and leaves it for good,
        never inside/outside/inside/outside. Every design this module
        builds is (petals and grid cells all reach outward from a compact
        core); it is the caller's job if a hand-built design is not.

        A ray that finds no interior point at all (the whole design offset
        away from its own bbox centre, say) falls back to the average of
        its neighbours' radii and a warning -- there is no sound answer for
        a non-star-shaped design short of a full 2-D silhouette algorithm,
        which is out of scope here.
        """
        u0, u1, v0, v1 = self.bbox
        cu, cv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
        r_max = 1.05 * np.hypot(max(u1 - cu, cu - u0), max(v1 - cv, cv - v0))

        def inside(u, v):
            out = None
            for f in self.facets:
                hit = f.region.contains(u - f.offset_mm[0], v - f.offset_mm[1])
                out = hit if out is None else (out | hit)
            return out

        angles = 2.0 * np.pi * np.arange(n_vertices) / n_vertices
        radii = np.full(n_vertices, np.nan)
        n_scan = 512
        scan_r = np.linspace(0.0, r_max, n_scan)

        for i, a in enumerate(angles):
            c, s = np.cos(a), np.sin(a)
            hits = inside(cu + scan_r * c, cv + scan_r * s)
            idx = np.nonzero(hits)[0]
            if idx.size == 0:
                continue  # left as nan; filled in below
            last = int(idx[-1])
            lo = float(scan_r[last])
            hi = float(scan_r[last + 1]) if last + 1 < n_scan else r_max
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if bool(inside(cu + mid * c, cv + mid * s)):
                    lo = mid
                else:
                    hi = mid
            radii[i] = lo

        missing = np.isnan(radii)
        if missing.any():
            if missing.all():
                raise ValueError(
                    "silhouette: no interior point found on any radial ray -- "
                    "this design is not star-shaped about its bbox centre"
                )
            warnings.warn(
                f"silhouette: {int(missing.sum())} of {n_vertices} azimuth(s) had no "
                "interior point on the radial ray from the design centre; falling back "
                "to the average of neighbouring vertices. silhouette() assumes the "
                "design is star-shaped about its centre.",
                stacklevel=2,
            )
            good = np.where(~missing)[0]
            for i in np.where(missing)[0]:
                left = good[good < i]
                right = good[good > i]
                left_r = radii[left[-1]] if left.size else radii[good[-1]]
                right_r = radii[right[0]] if right.size else radii[good[0]]
                radii[i] = 0.5 * (left_r + right_r)

        u = cu + radii * np.cos(angles)
        v = cv + radii * np.sin(angles)
        return Polygon(np.column_stack([u, v]))

    # -- preview --------------------------------------------------------

    def preview(self, ax=None, show_silhouette: bool = True, resolution: int = 600):
        """Render reflective facet fill against the :meth:`silhouette` outline.

        The picture this exists for: wherever the dashed silhouette line
        encloses area the solid fill does not reach, that area shades a
        neighbour without reflecting anything itself -- a facet gap, or the
        backing structure between petals. The filled facets are what an
        engine traces rays against; the silhouette is what a neighbouring
        heliostat's shading calculation should treat as opaque.

        Rasterises combined facet membership over the design bbox (+5%
        margin) on a ``resolution x resolution`` grid and imshows it as a
        two-level (transparent outside / solid steel-blue inside) image.
        With more than one facet, each facet's own membership edge is
        additionally drawn as a thin grey contour, so facet-to-facet joins
        are visible even where two facets abut with no gap. Returns the
        :class:`~matplotlib.axes.Axes` (``ax`` if given, else a new ~(6, 6)
        figure's).
        """
        # Lazy import: matplotlib is a declared dependency but importing it
        # at module load time would slow every caller of this module,
        # including ones that never plot anything.
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        u0, u1, v0, v1 = self.bbox
        margin_u = (u1 - u0) * 0.05 or (v1 - v0) * 0.05 or 1.0
        margin_v = (v1 - v0) * 0.05 or margin_u
        eu0, eu1, ev0, ev1 = u0 - margin_u, u1 + margin_u, v0 - margin_v, v1 + margin_v

        uu = np.linspace(eu0, eu1, resolution)
        vv = np.linspace(ev0, ev1, resolution)
        grid_u, grid_v = np.meshgrid(uu, vv)

        facet_masks = []
        combined = np.zeros(grid_u.shape, dtype=bool)
        for f in self.facets:
            mask = f.region.contains(grid_u - f.offset_mm[0], grid_v - f.offset_mm[1])
            facet_masks.append(mask)
            combined |= mask

        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 6.0))

        steel_blue = (0.27, 0.45, 0.62, 0.55)
        fill_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), steel_blue])
        ax.imshow(
            combined.astype(float),
            extent=(eu0, eu1, ev0, ev1),
            origin="lower",
            cmap=fill_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        if len(self.facets) > 1:
            for mask in facet_masks:
                ax.contour(
                    grid_u, grid_v, mask.astype(float), levels=[0.5], colors="0.5", linewidths=0.7
                )

        legend_handles = [
            Patch(facecolor=steel_blue, edgecolor="none", label="reflective aperture")
        ]
        if show_silhouette:
            poly = self.silhouette(72)
            verts = poly.vertices_mm
            closed = np.vstack([verts, verts[:1]])
            (line,) = ax.plot(
                closed[:, 0],
                closed[:, 1],
                linestyle="--",
                color="C1",
                linewidth=1.8,
                label="shading silhouette",
            )
            legend_handles.append(line)

        ax.set_aspect("equal")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_facecolor("white")
        ax.figure.patch.set_facecolor("white")
        ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)
        ax.figure.tight_layout()
        return ax


# ---------------------------------------------------------------------------
# canting


def cant_on_axis(facets: list[Facet], focal_mm: float) -> list[Facet]:
    """New facets, canted so an on-axis ray reflects centre -> ``(0, 0, focal_mm)``.

    "On-axis ray" means a ray travelling along the heliostat's own ``-n``
    (i.e. arriving from the ``+n`` side, direction ``(0, 0, -1)`` in the
    heliostat frame) that strikes a facet dead centre. The returned
    ``cant_normal`` is the bisector of that incoming ray reversed and the
    outgoing direction toward the focal point -- the usual mirror-normal
    construction, verified numerically in the test suite (reflect ``(0, 0,
    -1)`` off ``cant_normal`` from the facet centre and check it lands on
    ``(0, 0, focal_mm)``).
    """
    up = np.array([0.0, 0.0, 1.0])
    out = []
    for f in facets:
        ou, ov = f.offset_mm
        target_dir = np.array([-ou, -ov, focal_mm], dtype=float)
        target_dir /= np.linalg.norm(target_dir)
        cant_normal = target_dir + up
        cant_normal /= np.linalg.norm(cant_normal)
        out.append(replace(f, cant_normal=cant_normal))
    return out


def _resolve_focal(
    surface: Surface, offset_mm: tuple[float, float], cant_focal_mm: float | None
) -> Surface:
    """Turn ``Spherical(focal_mm="slant")`` into this facet's true slant distance.

    The slant focal length is the facet centre's actual 3-D distance to the
    on-axis focal point ``(0, 0, cant_focal_mm)`` rather than the nominal
    on-axis focal length -- the standard per-facet correction so a facet far
    from the axis is figured for the range it actually sees. A no-op for
    anything that is not an unresolved ``Spherical``.
    """
    if isinstance(surface, Spherical) and isinstance(surface.focal_mm, str):
        if surface.focal_mm != "slant":
            raise ValueError(f"unknown Spherical.focal_mm placeholder {surface.focal_mm!r}")
        if cant_focal_mm is None:
            raise ValueError("Spherical(focal_mm='slant') needs cant_focal_mm to resolve against")
        ou, ov = offset_mm
        slant = float(np.hypot(np.hypot(ou, ov), cant_focal_mm))
        return replace(surface, focal_mm=slant)
    return surface


# ---------------------------------------------------------------------------
# builders


def rect_heliostat(
    width_mm: float = 5000.0, height_mm: float = 3000.0, surface: Surface | None = None
) -> HeliostatDesign:
    """A single flat rectangular facet -- the parity anchor.

    No cant, default figure ``ZernikeAstig(0, 0, 0)`` (flat): an engine that
    consumes this design and reproduces :mod:`heliostat.trace.mc`'s
    single-mirror trace exactly is the baseline every fancier design is
    checked against.
    """
    if surface is None:
        surface = ZernikeAstig(0.0, 0.0, 0.0)
    facet = Facet(
        region=Rect(width_mm, height_mm), surface=surface, offset_mm=(0.0, 0.0), cant_normal=None
    )
    return HeliostatDesign([facet])


def custom_heliostat(vertices_mm, surface: Surface | None = None) -> HeliostatDesign:
    """A single hand-drawn polygon facet -- the sketch-tool analogue of
    :func:`rect_heliostat`.

    ``vertices_mm`` is an ``(N>=3, 2)`` array-like of ``(u, v)`` corners in
    the heliostat's own plane, in order around the perimeter (either
    winding -- :class:`~heliostat.geometry.aperture.Polygon`'s crossing-
    number membership test does not care). No cant: like a rectangle, a
    single facet has no other facet to aim relative to.
    """
    if surface is None:
        surface = ZernikeAstig(0.0, 0.0, 0.0)
    facet = Facet(
        region=Polygon(vertices_mm), surface=surface, offset_mm=(0.0, 0.0), cant_normal=None
    )
    return HeliostatDesign([facet])


def grid_facets(
    n_u: int,
    n_v: int,
    facet_w_mm: float,
    facet_h_mm: float,
    gap_mm: float = 0.0,
    surface: Surface | None = None,
    cant_focal_mm: float | None = None,
) -> HeliostatDesign:
    """An ``n_u`` x ``n_v`` grid of rectangular facets, optionally canted on-axis."""
    if surface is None:
        surface = ZernikeAstig(0.0, 0.0, 0.0)
    pitch_u = facet_w_mm + gap_mm
    pitch_v = facet_h_mm + gap_mm
    total_w = n_u * facet_w_mm + (n_u - 1) * gap_mm
    total_h = n_v * facet_h_mm + (n_v - 1) * gap_mm

    facets = []
    for j in range(n_v):
        for i in range(n_u):
            ou = -total_w / 2.0 + facet_w_mm / 2.0 + i * pitch_u
            ov = -total_h / 2.0 + facet_h_mm / 2.0 + j * pitch_v
            fsurf = _resolve_focal(surface, (ou, ov), cant_focal_mm)
            facets.append(
                Facet(region=Rect(facet_w_mm, facet_h_mm), surface=fsurf, offset_mm=(ou, ov))
            )

    if cant_focal_mm is not None:
        facets = cant_on_axis(facets, cant_focal_mm)
    return HeliostatDesign(facets)


def _petal_at_angle(length_mm: float, width_mm: float, theta_deg: float) -> Region:
    """A :func:`~heliostat.geometry.aperture.petal` forward-rotated by ``theta_deg``.

    Built from :class:`~heliostat.geometry.aperture.Disc`/``Translate``/
    ``Intersection`` directly (the same two-circle-lens recipe
    :func:`~heliostat.geometry.aperture.petal` uses, with both disc centres
    and the tip pre-rotated) rather than via ``petal(...).rotated(theta_deg)``.

    That avoids a real bug found while building this module:
    ``aperture.Rotate.bbox()`` rotates the child's corners by ``-theta_deg``
    instead of ``+theta_deg`` -- the opposite sign from
    ``aperture.Rotate.contains()``, which is internally consistent (checked
    numerically: :func:`petal` rotated this way agrees with
    ``petal(...).rotated(theta_deg)`` on membership for every angle tested,
    to the last bit). The bug is invisible for shapes centred on the origin
    (``Rect``, and any single ``CircularArray`` that sweeps a full 360°,
    which happens to cancel it out by symmetry) but silently corrupts both
    ``bbox()`` and the default numeric ``area_mm2()`` for any *other*
    off-centre rotated shape -- exactly this module's per-petal facets.
    Flagged upstream rather than patched here, since ``aperture.py`` is
    existing code out of scope for this change.
    """
    if width_mm >= 2.0 * length_mm:
        raise ValueError("petal width must be less than twice its length")
    half_l = length_mm / 2.0
    half_w = width_mm / 2.0
    r = (half_l * half_l + half_w * half_w) / (2.0 * half_w)
    a = np.deg2rad(theta_deg)
    c, s = np.cos(a), np.sin(a)

    def fwd(dx, dy):
        return (c * dx - s * dy, s * dx + c * dy)

    d = r - half_w
    lens = Disc(r).translated(*fwd(d, 0.0)) & Disc(r).translated(*fwd(-d, 0.0))
    return lens.translated(*fwd(0.0, half_l))


def flower(
    n_petals: int = 5,
    petal_length_mm: float = 2000.0,
    petal_width_mm: float = 900.0,
    hub_radius_mm: float = 0.0,
    surface: Surface | None = None,
    cant_focal_mm: float | None = None,
    petals_as_facets: bool = True,
) -> HeliostatDesign:
    """A ``n_petals``-petal flower mirror.

    ``petals_as_facets=True`` gives one :class:`Facet` per petal, each in
    its own local frame rotated so ``+lv`` points outward along that petal
    (baked into the facet's ``region`` via :func:`_petal_at_angle`, since
    ``Facet`` itself has no in-plane rotation field -- see the module
    docstring).
    ``petals_as_facets=False`` gives a single facet whose region is the
    :func:`~heliostat.geometry.aperture.circular_array` sketch used by the
    fixture examples; the two must cover the same footprint (checked in the
    test suite).
    """
    if surface is None:
        surface = ZernikeAstig(0.0, 0.0, 0.0)

    if not petals_as_facets:
        sketch = circular_array(
            petal(petal_length_mm, petal_width_mm).translated(0.0, hub_radius_mm), n_petals
        )
        facets = [Facet(region=sketch, surface=surface, offset_mm=(0.0, 0.0))]
        if cant_focal_mm is not None:
            facets = cant_on_axis(facets, cant_focal_mm)
        return HeliostatDesign(facets)

    step = 360.0 / n_petals
    facets = []
    for i in range(n_petals):
        theta_deg = i * step
        theta = np.deg2rad(theta_deg)
        offset = (-hub_radius_mm * np.sin(theta), hub_radius_mm * np.cos(theta))
        region = _petal_at_angle(petal_length_mm, petal_width_mm, theta_deg)
        fsurf = _resolve_focal(surface, offset, cant_focal_mm)
        facets.append(Facet(region=region, surface=fsurf, offset_mm=offset))

    if cant_focal_mm is not None:
        facets = cant_on_axis(facets, cant_focal_mm)
    return HeliostatDesign(facets)
