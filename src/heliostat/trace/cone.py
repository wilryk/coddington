"""Cone-optics trace: deterministic rays plus analytic sunshape deposition.

The Monte Carlo backend's noise comes from sampling the sun's angular
distribution with random rays; the distribution itself is known exactly.
This backend samples the *mirror surface* on a small deterministic grid
instead. Each sample point contributes five rays: a chief ray from the
sun's centre, and four rays perturbed by a small angle along two axes
perpendicular to the sun vector. The chief ray's landing point gives the
kernel centre; the four companions give the local Jacobian
``d(receiver uv)/d(source angle)`` by central differences. The sun's
angular density — optionally broadened by Gaussian slope and tracking
error — is then laid down through that Jacobian by
:func:`heliostat.trace.kernels.deposit`.

Because the Jacobian is *measured* through the real optical chain, every
geometric effect the Monte Carlo trace captures — off-axis astigmatism,
cone folding, hyperboloid magnification, receiver obliquity — is inherited
without being modelled.

Edges — secondary rims, receiver-window borders, neighbour shadows and
neighbour blocking — are handled in angle space, where they actually
live: every sample's transmission is *measured* on a ``mask_nodes``² node
grid spanning the kernel support (one vectorised bundle for the whole
mirror), and partially-clipped kernels deposit through the resulting
raster, penumbra included. Samples whose kernel centre is itself lost at
a rim deposit their surviving mass directly at the passing nodes'
landing points instead of being dropped.

A 20 x 12 grid (240 samples) reproduces the smooth flux structure that
~10^5 Monte Carlo rays estimate; error falls with grid and node density,
not with luck.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry.receiver import Receiver
from ..geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    Secondary,
    secondary_bin_areas_m2,
    secondary_has_flux_map,
    secondary_uv,
    secondary_uv_extent,
)
from ..geometry.shading import _blocked_mask
from .bspline_deposit import control_grid_edges, evaluate_bspline
from .kernels import RadialKernel, deposit
from .mc import (
    MIRROR_HALF_X_MM,
    MIRROR_HALF_Y_MM,
    _mirror_frame,
    _sun_vector,
    _zernike_sag_and_slopes,
    design_facet_frames,
)
from .samplers import BUIE_LIMB_MRAD, SUPER_GAUSS_ORDER, SUPER_GAUSS_SIGMA_RAD

STANDARD_IRRADIANCE_W_MM2 = 1000.0e-6  # 1000 W/m^2, the trace normalisation

# Skip-test safety cushions (see `_reach_mm`/`_secondary_ring_clears`): both
# bounds are good estimates of the true footprint, not airtight proofs, so
# each is required to clear its boundary by more than just zero.
WINDOW_SAFETY_FACTOR = 1.25  # relative cushion on the receiver-window reach
WINDOW_MARGIN_FLOOR_MM = 5.0  # absolute floor for a near-zero reach
SECONDARY_MARGIN_FRAC = 0.01  # relative cushion on the secondary aperture
SECONDARY_MARGIN_FLOOR_MM = 20.0  # absolute floor for a small aperture
RING_PROBES = 12  # boundary directions probed per sample for the rim check

# deposit_method="bspline" (see trace_heliostat_cone's docstring):
# control_grid=None derives a coarse accumulation grid this many times
# smaller than flux_grid, per axis (scripts/coeff_prototype/REPORT.md's own
# 32x32-vs-128x128 benchmark is a 4x coarsening), clamped to a minimum so a
# very small flux_grid still gets a usable control grid.
CONTROL_GRID_COARSEN = 4
CONTROL_GRID_MIN = 8

#: Test-only escape hatch: forces every sample through the full node raster,
#: bypassing the skip test below, so a test can compare "skip enabled" against
#: "skip forced off" on the same geometry. Never set outside a test.
DISABLE_TRANSMISSION_SKIP = False


#: Tail mass fraction treated as negligible when tightening the skip test's
#: footprint radius below the kernel's own (deliberately generous)
#: ``support_rad`` -- two orders of magnitude under the ``deposit`` full_pass
#: threshold of ``1e-9`` (see ``_effective_support_rad``), so even a worst
#: case where every bit of that tail were clipped cannot flip a skip result.
SKIP_TAIL_MASS_TOL = 1.0e-11


def _effective_support_rad(kernel: RadialKernel, tail_tol: float = SKIP_TAIL_MASS_TOL) -> float:
    """Angular radius beyond which ``kernel``'s remaining mass is provably
    below ``tail_tol`` of its total.

    ``kernel.support_rad`` is simply where the tabulated profile ends --
    for a smooth sunshape that is deep in a negligible tail, and using it
    directly makes the skip test's footprint bound needlessly wide. This
    integrates the kernel's own tabulated density (its ``__init__`` already
    normalises ``2*pi*integral(density*theta, theta) == 1``) to find the
    smallest radius whose remaining tail mass is negligible even against
    ``deposit``'s ``1e-9`` full_pass threshold, then uses only that radius
    for the skip test -- ``deposit`` itself is untouched and keeps using
    the kernel's full support.
    """
    theta = kernel.theta_rad
    mass = 2.0 * np.pi * kernel.density * theta
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (mass[1:] + mass[:-1]) * np.diff(theta))])
    tail = cum[-1] - cum
    within_tol = tail <= tail_tol * cum[-1]
    if not np.any(within_tol):
        return kernel.support_rad
    return float(theta[np.argmax(within_tol)])


def _jac_smax_mm(jac_all: np.ndarray) -> np.ndarray:
    """Vectorised top singular value of each sample's local Jacobian.

    Shared by the transmission-skip test's footprint-reach bound
    (:func:`_reach_mm`) and :func:`~heliostat.trace.kernels.deposit`'s own
    bounding-box sizing, which needs the identical per-sample value — this
    computes it once for all ``m`` samples instead of once more per sample
    inside ``deposit``'s Python loop. Samples without a Jacobian (NaN)
    get a harmless zero-filled input and thus a zero smax; callers gate
    those out separately.
    """
    jac_safe = np.where(np.isnan(jac_all), 0.0, jac_all)
    jjt = np.einsum("mij,mkj->mik", jac_safe, jac_safe)
    return np.sqrt(np.clip(np.linalg.eigvalsh(jjt)[:, -1], 0.0, None))


def _reach_mm(
    smax: np.ndarray, hess_all: np.ndarray | None, full_stencil: np.ndarray, support_rad: float
) -> np.ndarray:
    """Footprint-reach bound, one value per sample: ``smax`` (the top
    singular value of the local Jacobian, from :func:`_jac_smax_mm`) times
    the kernel support radius, plus the order-2 Hessian correction where
    one was measured (see :func:`~heliostat.trace.kernels.deposit` for the
    derivation of both terms). Samples without a Jacobian carry a harmless
    zero ``smax``; callers gate those out separately.
    """
    reach = smax * support_rad
    if hess_all is not None:
        hmax = np.zeros(smax.shape[0])
        hmax[full_stencil] = np.abs(hess_all[full_stencil]).max(axis=(1, 2, 3))
        reach = reach + hmax * support_rad**2
    return reach


def _secondary_ring_clears(
    pts: np.ndarray,
    normal: np.ndarray,
    s: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    support_rad: float,
    secondary,
    margin_mm: float,
    n_ring: int,
) -> np.ndarray:
    """True per sample if every ray at the kernel's angular *boundary*
    lands on ``secondary`` within its aperture, less ``margin_mm``.

    Reflecting the sun cone's boundary circle (radius ``support_rad`` around
    ``-s``) off a sample's fixed mirror point and normal is an isometry --
    it maps onto exactly the boundary circle of the outgoing ray cone
    around the sample's chief reflected direction, so these probes are
    exact rays through the real secondary, not a linear extrapolation.
    Testing only the boundary (not the interior) bounds the whole disk for
    a surface whose hit-radius has no interior maximum strictly inside the
    disk -- true for these axisymmetric conics away from grazing
    incidence; ``margin_mm`` is the cushion against a finite ring count and
    against that assumption. A probe that misses ``secondary`` outright
    scores an infinite radius, so it always fails the margin.
    """
    m = pts.shape[1]
    psi = 2.0 * np.pi * np.arange(n_ring) / n_ring
    au = support_rad * np.cos(psi)
    av = support_rad * np.sin(psi)
    d_in = -s[:, None] + au[None, :] * e1[:, None] + av[None, :] * e2[:, None]
    d_in /= np.linalg.norm(d_in, axis=0, keepdims=True)  # (3, n_ring)

    dots = normal.T @ d_in  # (m, n_ring)
    d_out = d_in[:, None, :] - 2.0 * dots[None, :, :] * normal[:, :, None]  # (3, m, n_ring)
    d_out_flat = d_out.reshape(3, m * n_ring).copy()
    p_flat = np.repeat(pts, n_ring, axis=1)

    hit_pt, _, on_sec = secondary.redirect(p_flat, d_out_flat, {})
    local_hit = secondary.to_local_point(hit_pt)
    radial = np.full(m * n_ring, np.inf)
    radial[on_sec] = np.hypot(local_hit[0], local_hit[1])
    worst = radial.reshape(m, n_ring).max(axis=1)
    return worst <= (secondary.aperture_radius_mm - margin_mm)


def sunshape_kernel(
    source_model: str = "super_gauss",
    slope_error_mrad: float = 0.0,
    pointing_error_mrad: float = 0.0,
    specularity_mrad: float = 0.0,
) -> RadialKernel:
    """The angular kernel for a named sunshape, optionally error-broadened.

    Profiles are the same pinned forms the Monte Carlo samplers draw from.
    Mirror slope error deflects a reflected ray by twice the surface tilt,
    hence the factor 2 on ``slope_error_mrad``; ``specularity_mrad`` is a
    coating scatter of the reflected ray itself, so it carries no such
    doubling -- the same distinction (and the same two error sources) the
    Monte Carlo backend's ``trace_heliostat`` applies per-ray.

    ``pointing_error_mrad`` (docs/ui-spec-v0.2.md §F) is the tracker's
    aiming inaccuracy, folded in as a third broadening term alongside the
    two above -- "the annual energy hit of a sloppy tracker shows up at
    every fidelity" (spec §F), even though at cone fidelity there is no
    per-instant "misses left/misses right" character to show, only its
    long-run, ensemble-averaged effect as an added broadening (equivalent,
    in variance, to convolving the base kernel with the same Gaussian that
    Monte Carlo's many independent per-timestep offsets would average out
    to over many instants -- see ``heliostat.trace.mc.trace_heliostat``'s
    own ``pointing_error_mrad`` docstring for that MC side). By the
    resolved spec convention, ``pointing_error_mrad`` is already the RMS
    angular deviation of the REFLECTED beam, not the mirror tilt that
    produces it -- so unlike ``slope_error_mrad`` (a mirror-tilt RMS that
    the reflection law doubles), it carries NO factor of two here: it
    enters the ``hypot`` below exactly as given, the same convention
    ``specularity_mrad`` already uses (also a reflected-beam-frame
    quantity). Getting this backwards -- doubling ``pointing_error_mrad``
    the way ``slope_error_mrad`` is doubled -- would silently broaden the
    cone spot by 2x what a matching Monte Carlo trace realises.
    """
    if source_model == "super_gauss":
        sig = SUPER_GAUSS_SIGMA_RAD

        def profile(t):
            return np.exp(-((t**2 / (2.0 * sig**2)) ** SUPER_GAUSS_ORDER))

        # Support 4.5 sigma: the order-2 super-Gaussian falls as
        # exp(-(theta^2/2sigma^2)^2), ~1e-45 there. A generous-looking 8
        # sigma support would make every footprint's bounding box overlap
        # the grid edge and disable per-kernel mass renormalisation.
        kernel = RadialKernel.from_profile(profile, support_rad=4.5 * sig)
    elif source_model == "buie":
        limb = BUIE_LIMB_MRAD * 1e-3

        def profile(t):
            t_mrad = np.minimum(t, limb) * 1e3
            return np.where(t <= limb, np.cos(0.326 * t_mrad) / np.cos(0.308 * t_mrad), 0.0)

        kernel = RadialKernel.from_profile(profile, support_rad=limb)
    else:
        raise ValueError(f"unknown source_model {source_model!r}")

    broadening = (
        np.hypot(np.hypot(2.0 * slope_error_mrad, specularity_mrad), pointing_error_mrad) * 1e-3
    )
    return kernel.convolve_gaussian(broadening) if broadening > 0 else kernel


def grid_for_density(
    density_per_m2: float, width_m: float, height_m: float, min_n: int = 2
) -> tuple[int, int]:
    """``(n_x, n_y)`` mirror-sample grid for ``density_per_m2`` samples/m^2
    over a ``width_m x height_m`` aperture bounding box, aspect-matched to
    that bbox.

    Requiring both ``n_x * n_y == density * width_m * height_m`` (the
    requested sample count) and ``n_x / n_y == width_m / height_m`` (aspect
    matched to the bbox) simultaneously gives, by substitution, ``n_x ==
    width_m * sqrt(density)`` and ``n_y == height_m * sqrt(density)`` -- each
    axis independently its own physical length times ``sqrt(density)``,
    rounded (floored at ``min_n`` per axis). On the manuscript's 5m x 3m
    mirror at ``density=12.0`` this gives exactly ``(17, 10)``; at the old
    hardcoded grid's own implied density (``16.0``) it reproduces ``(20,
    12)`` exactly. See ``scripts/coeff_prototype/REPORT.md`` SS7 for the
    sparsity sweep (full 643-heliostat field) that picked density=12.0 as
    the sparsest rung holding field-total error under ~0.1% vs the
    hardcoded grid.
    """
    s = math.sqrt(density_per_m2)
    n_x = max(min_n, round(width_m * s))
    n_y = max(min_n, round(height_m * s))
    return n_x, n_y


def trace_heliostat_cone(
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
    kernel: RadialKernel,
    grid: tuple[int, int] | None = (20, 12),
    density: float | None = None,
    flux_grid: tuple[int, int] = (128, 128),
    delta_rad: float = 2.0e-4,
    order: int = 1,
    occluders: list | None = None,
    shadow_body=None,
    mask_nodes: int = 16,
    design=None,
    return_secondary_flux: bool = False,
    secondary_flux_grid: tuple[int, int] = (128, 128),
    deposit_method: str = "binned",
    control_grid: tuple[int, int] | None = None,
) -> dict:
    """Cone-optics trace of one heliostat at one instant.

    Same pointing, figure, secondary and receiver conventions as
    :func:`heliostat.trace.mc.trace_heliostat` (including the c4/c5 frame
    correction), so results are directly comparable at identical inputs.
    Returns a dict with ``flux`` (``(n_v, n_u)`` W/m² at the 1000 W/m²
    trace normalisation), ``power_w`` (its integral over the receiver
    window), ``incident_power_w`` (cosine-weighted power arriving on the
    mirror), and a counter chain ``samples / valid / blocked / unresolved``
    over mirror sample cells.

    ``grid`` is the mirror-surface sample grid, ``(n_x, n_y)``. Passing
    ``grid=None`` instead resolves it from ``density`` (samples/m^2, required
    in that case) via :func:`grid_for_density`, aspect-matched to the
    mirror's own aperture bbox (``MIRROR_HALF_X_MM``/``MIRROR_HALF_Y_MM`` for
    a plain rectangle, ``design.bbox`` for a custom design) rather than a
    fixed tuple baked to the manuscript's 5m x 3m mirror. This is how the
    ``ultra_fast`` mode (:mod:`heliostat.trace.modes`) samples; every other
    caller keeps passing an explicit ``grid`` tuple and is unaffected.

    ``deposit_method`` selects how sample kernels are laid down onto the
    receiver window. ``"binned"`` (default) deposits every sample directly
    onto the fine ``flux_grid`` via :func:`heliostat.trace.kernels.deposit`
    -- exact per :mod:`heliostat.trace.kernels`'s own analysis, cost scales
    with footprint size on the fine grid. ``"bspline"`` instead accumulates
    onto a coarse ``control_grid`` spanning the same window -- "binning with
    smooth bins," the *same* ``deposit`` call, just far fewer cells per
    footprint -- then upsamples once, at the end, via a fixed cubic-B-spline
    interpolation matrix onto the requested ``flux_grid``; see
    :mod:`heliostat.trace.bspline_deposit` for the accumulate/evaluate math
    and the cylinder-seam periodicity note. ``control_grid=None`` (default)
    derives it as ``flux_grid`` coarsened ``CONTROL_GRID_COARSEN`` (4x) per
    axis rather than a fixed tuple: a curved receiver's adaptive
    ``flux_grid`` (``_receiver_flux_grid`` in the web layer scales ``n_u``
    up to 448 for a wide cylinder) needs a proportionally wider control grid
    too, or the fixed 32x32 the prototype benchmarked at flat 128x128 becomes
    a far coarser-than-intended ~14x coarsening instead of 4x, badly
    under-resolving peak flux. This is how the ``ultra_fast`` mode
    (:mod:`heliostat.trace.modes`) deposits; ``fast_accurate`` and Monte
    Carlo keep exact binned deposit.

    ``delta_rad`` is the finite-difference probe angle; it must be small
    against the optics' scale of nonlinearity but large enough that
    receiver-position differences dominate roundoff — anything within an
    order of magnitude of the default works for metre-scale optics.

    ``order`` selects the local model of the optical map. ``1`` (five rays
    per sample) linearises it — the ultra-fast mode, leaving a curvature
    residual of ~1% of peak in the flux map. ``2`` (nine rays per sample)
    also measures the Hessian by finite differences and deposits through
    the quadratic map, removing that leading error term for roughly twice
    the cost — the fast-and-accurate mode.

    Edges and occlusion. A sample whose kernel straddles any boundary —
    secondary rim, receiver-window edge, a neighbour's shadow on the
    incoming side, a neighbour blocking the outgoing beam — gets a
    ``mask_nodes``² transmission raster in angle space, built by testing
    that many deterministic node rays from the same surface point
    (mini-trace through secondary and receiver; ray-vs-rectangle tests
    against ``occluders``; ``shadow_body.occludes`` for an opaque
    secondary body). The kernel is deposited through the raster, so edges
    are resolved with true penumbra rather than at sample granularity.
    ``occluders`` is this heliostat's neighbour list as
    :class:`~heliostat.geometry.shading.MirrorGeometry` rectangles; leave
    it ``None`` for store-bound sweeps, where the store contract applies
    occlusion as read-time scalars instead. Counter invariant:
    ``valid + masked + blocked + node_fallback + unresolved == samples``.

    ``return_secondary_flux`` (default ``False``, so a caller who never asks
    for it gets bit-identical receiver-path results) additionally computes
    an incident-flux map on the secondary's own surface, when ``secondary``
    has one (:func:`~heliostat.geometry.secondary.secondary_has_flux_map` --
    axicon/Cassegrain only; silently omitted for :class:`NoSecondary` or a
    :class:`~heliostat.geometry.secondary.PyramidSecondary`). Unlike the
    receiver deposit, this is deliberately COARSE: each sample deposits its
    full weight at its CHIEF ray's secondary hit point (no footprint/
    Jacobian spread onto the secondary's own surface -- a second Jacobian in
    the secondary's ``(u, v)`` is a real refinement left for later), with
    samples whose chief ray misses the secondary rim falling back to
    depositing at each surviving node's own hit point, mirroring the
    receiver deposit's ``node_fallback`` path. Exact per-ray accounting is
    what :func:`heliostat.trace.mc.trace_heliostat`'s
    ``return_secondary_hits=True`` gives instead -- "coarse in cone modes,
    exact in Monte Carlo" is the fidelity disclosure the UI must carry
    wherever this map is shown. Adds ``secondary_flux``, ``secondary_u_edges``,
    ``secondary_v_edges``, ``secondary_power_w`` and ``secondary_fidelity``
    to the returned dict when computed.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    if deposit_method not in ("binned", "bspline"):
        raise ValueError(f"deposit_method must be 'binned' or 'bspline', got {deposit_method!r}")
    # The order-2 deposit spreads each sample by a Hessian of the map from
    # ray angle to surface position. That map folds on a curved receiver,
    # and a fold drives the deposit's density factor through zero: the cap
    # in kernels.py keeps total power honest there, but peak flux does not
    # survive it. Curved surfaces take the linear deposit.
    if not getattr(receiver, "is_planar", True):
        order = 1
    # Same frame bookkeeping as the MC trace: the figure coefficients
    # arrive in a convention whose y/z flip negates c4 and c5 here.
    c4 = -c4
    c5 = -c5

    s = _sun_vector(solar_az_deg, solar_el_deg)
    helio = np.array([x_mm, y_mm, 0.0])
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), s)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)

    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)

    # Mirror-surface sample grid: cell centres, area-weighted. `grid=None`
    # resolves to a density-derived grid instead of a fixed tuple -- see
    # `grid_for_density` and the docstring above.
    if grid is None:
        if density is None:
            raise ValueError("grid=None requires density (samples/m^2) to resolve a grid")
        if design is None:
            bbox_width_m = 2.0 * MIRROR_HALF_X_MM / 1000.0
            bbox_height_m = 2.0 * MIRROR_HALF_Y_MM / 1000.0
        else:
            bbox_du0, bbox_du1, bbox_dv0, bbox_dv1 = design.bbox
            bbox_width_m = (bbox_du1 - bbox_du0) / 1000.0
            bbox_height_m = (bbox_dv1 - bbox_dv0) / 1000.0
        n_x, n_y = grid_for_density(density, bbox_width_m, bbox_height_m)
    else:
        n_x, n_y = grid
    if design is None:
        gx = (np.arange(n_x) + 0.5) / n_x * 2.0 * MIRROR_HALF_X_MM - MIRROR_HALF_X_MM
        gy = (np.arange(n_y) + 0.5) / n_y * 2.0 * MIRROR_HALF_Y_MM - MIRROR_HALF_Y_MM
        lx, ly = (a.ravel() for a in np.meshgrid(gx, gy))
        m = lx.size
        cell_area_mm2 = (2.0 * MIRROR_HALF_X_MM / n_x) * (2.0 * MIRROR_HALF_Y_MM / n_y)
        area_w = np.full(m, cell_area_mm2)

        sag, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
        pts = helio[:, None] + u[:, None] * lx + v[:, None] * ly + n[:, None] * sag  # (3, M)
        normal = n[:, None] - u[:, None] * dsdx - v[:, None] * dsdy
        normal /= np.linalg.norm(normal, axis=0)
    else:
        # Per-facet cell grids at a uniform cell size derived from the
        # design's bbox, cells kept with their membership fraction
        # (4x4 sub-sampled) so sketch boundaries carry fractional area
        # instead of stair-stepping.
        du0, du1, dv0, dv1 = design.bbox
        cell_w = (du1 - du0) / n_x
        cell_h = (dv1 - dv0) / n_y
        pts_list, nrm_list, area_list = [], [], []
        sub = (np.arange(4) + 0.5) / 4.0 - 0.5  # coarse cell-relative sub-offsets
        sub_u, sub_v = (a.ravel() for a in np.meshgrid(sub, sub))
        fine = (np.arange(16) + 0.5) / 16.0 - 0.5  # refinement for boundary cells
        fine_u, fine_v = (a.ravel() for a in np.meshgrid(fine, fine))
        frames_list = design_facet_frames(design, helio, n, u, v)
        for k_idx, (facet, nf, fu, fv, centre) in enumerate(frames_list):
            b0, b1, c0, c1 = facet.region.bbox()
            k_u = max(1, int(np.ceil((b1 - b0) / cell_w)))
            k_v = max(1, int(np.ceil((c1 - c0) / cell_h)))
            cu = b0 + (np.arange(k_u) + 0.5) * (b1 - b0) / k_u
            cv = c0 + (np.arange(k_v) + 0.5) * (c1 - c0) / k_v
            lu, lv = (a.ravel() for a in np.meshgrid(cu, cv))
            fw = (b1 - b0) / k_u
            fh = (c1 - c0) / k_v
            sub_lu = lu[:, None] + sub_u[None, :] * fw
            sub_lv = lv[:, None] + sub_v[None, :] * fh
            member = facet.region.contains(sub_lu, sub_lv)
            # Overlapping facets (petal bases at a small hub, say) must not
            # deposit the same mirror area twice: each patch of the
            # heliostat plane belongs to the FIRST facet covering it,
            # matching the MC path's nearest-intersection rule for the
            # near-coplanar overlaps a sane design can contain.
            ou, ov = facet.offset_mm
            for prev, *_ in frames_list[:k_idx]:
                member &= ~prev.region.contains(
                    sub_lu + (ou - prev.offset_mm[0]), sub_lv + (ov - prev.offset_mm[1])
                )
            frac = member.mean(axis=1)
            # Boundary cells (partial at the coarse screen) get a 16x16
            # refinement: thin sketches make most kept cells boundary
            # cells, and the coarse fraction over-counts curved edges by
            # ~0.5% of total area — enough to show up against MC.
            partial = (frac > 0.0) & (frac < 1.0)
            if np.any(partial):
                p_lu = lu[partial, None] + fine_u[None, :] * fw
                p_lv = lv[partial, None] + fine_v[None, :] * fh
                fmem = facet.region.contains(p_lu, p_lv)
                for prev, *_ in frames_list[:k_idx]:
                    fmem &= ~prev.region.contains(
                        p_lu + (ou - prev.offset_mm[0]), p_lv + (ov - prev.offset_mm[1])
                    )
                frac[partial] = fmem.mean(axis=1)
            keep = frac > 0.0
            lu, lv, frac = lu[keep], lv[keep], frac[keep]
            if lu.size == 0:
                continue
            sag, dsu, dsv = facet.surface.sag_and_slopes(lu, lv)
            pts_list.append(
                centre[:, None] + fu[:, None] * lu + fv[:, None] * lv + nf[:, None] * sag
            )
            nrm = nf[:, None] - fu[:, None] * dsu - fv[:, None] * dsv
            nrm_list.append(nrm / np.linalg.norm(nrm, axis=0))
            area_list.append(frac * fw * fh)
        pts = np.concatenate(pts_list, axis=1)
        normal = np.concatenate(nrm_list, axis=1)
        area_w = np.concatenate(area_list)
        m = pts.shape[1]

    # Incoming directions, identical for every sample (the sun is at
    # infinity). Stencil legs: [chief, +e1, -e1, +e2, -e2] and, at order 2,
    # the four diagonals [++, +-, -+, --] that resolve the mixed Hessian.
    stencil = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    if order == 2:
        stencil += [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    offsets = np.array(stencil) * delta_rad
    legs = offsets.shape[0]
    dirs = -s[None, :] + offsets[:, 0, None] * e1[None, :] + offsets[:, 1, None] * e2[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # (5, 3)

    # Assemble all legs*M rays, stencil-major: ray index k*m + i is stencil
    # leg k at sample i. Every leg starts at the sample point itself.
    p_flat = np.concatenate([pts] * legs, axis=1)
    normal5 = np.concatenate([normal] * legs, axis=1)
    d_flat = np.repeat(dirs.T, m, axis=1)  # (3, legs*M), stencil-major

    dot = 2.0 * np.einsum("ij,ij->j", d_flat, normal5)
    d_ref = d_flat - dot * normal5

    counters = {"samples": int(m)}
    pre, d_out, on_sec = secondary.redirect(p_flat, d_ref, {})
    hit_mask, uv_hits = receiver.intersect(pre, d_out)

    # Map survival back to (stencil, sample): redirect returns the filtered
    # survivor bundle plus its boolean mask over the input rays; intersect
    # filters again within the survivors.
    alive = np.zeros(legs * m, dtype=bool)
    survivors = np.flatnonzero(on_sec)
    alive[survivors[hit_mask]] = True
    uv = np.full((2, legs * m), np.nan)
    uv[:, survivors[hit_mask]] = uv_hits

    alive = alive.reshape(legs, m)
    uv = uv.reshape(2, legs, m)

    want_sec = return_secondary_flux and secondary_has_flux_map(secondary)
    if want_sec:
        # Chief leg's own secondary hit point, recovered the same way `uv`
        # is above: `pre` is already the (3, on_sec.sum()) world hit points
        # `secondary.redirect` computed unconditionally -- no new ray
        # tracing -- scattered back to (3, legs*m) via `survivors`, then
        # sliced to leg 0 (stencil-major layout: leg k occupies indices
        # [k*m, (k+1)*m)).
        sec_pts_flat = np.full((3, legs * m), np.nan)
        sec_pts_flat[:, survivors] = pre
        chief_sec_pts = sec_pts_flat.reshape(3, legs, m)[:, 0, :]
        chief_on_sec = on_sec.reshape(legs, m)[0]

    chief_ok = alive[0]
    full_stencil = alive.all(axis=0)

    # Jacobian per sample, mm per rad: central differences where both
    # partners of an axis survived, one-sided where only one did. A sample
    # deposits when its chief ray and at least one partner per axis exist.
    jac_all = np.full((m, 2, 2), np.nan)
    can_jac = chief_ok.copy()
    for axis, (leg_p, leg_m) in enumerate([(1, 2), (3, 4)]):
        both = alive[leg_p] & alive[leg_m]
        pos = alive[leg_p] & ~alive[leg_m] & chief_ok
        neg = alive[leg_m] & ~alive[leg_p] & chief_ok
        col = np.full((2, m), np.nan)
        col[:, both] = (uv[:, leg_p, both] - uv[:, leg_m, both]) / (2.0 * delta_rad)
        col[:, pos] = (uv[:, leg_p, pos] - uv[:, 0, pos]) / delta_rad
        col[:, neg] = (uv[:, 0, neg] - uv[:, leg_m, neg]) / delta_rad
        jac_all[:, :, axis] = col.T
        can_jac &= alive[leg_p] | alive[leg_m]

    # Top singular value of each sample's Jacobian, computed once for every
    # sample: shared below by the transmission-skip test's reach bound and
    # by `deposit`'s own bounding-box sizing (see `_jac_smax_mm`).
    smax_all = _jac_smax_mm(jac_all)

    hess_all = None
    if order == 2:
        # Second differences where the full stencil survived; a partial
        # stencil falls back to the linear deposit for that sample.
        d2 = delta_rad * delta_rad
        hess_all = np.full((m, 2, 2, 2), np.nan)
        f = full_stencil
        chief = uv[:, 0]
        hess_all[f, :, 0, 0] = ((uv[:, 1, f] - 2.0 * chief[:, f] + uv[:, 2, f]) / d2).T
        hess_all[f, :, 1, 1] = ((uv[:, 3, f] - 2.0 * chief[:, f] + uv[:, 4, f]) / d2).T
        mixed = (uv[:, 5, f] - uv[:, 6, f] - uv[:, 7, f] + uv[:, 8, f]) / (4.0 * d2)
        hess_all[f, :, 0, 1] = mixed.T
        hess_all[f, :, 1, 0] = mixed.T

    # --- angular transmission, measured on a node grid where it can't be
    # ruled out cheaply -----------------------------------------------------
    # The stencil spans only ~delta_rad and cannot detect a boundary lying
    # elsewhere inside the kernel's ~10 mrad support, so transmission is
    # not detected — it is measured: mask_nodes² node rays per sample, one
    # vectorised bundle for the whole mirror, through every clip the sun
    # cone can meet (neighbour shadow on the way in; neighbour blocking,
    # secondary aperture and receiver window on the way out).
    support = kernel.support_rad
    occluders = occluders or []
    (u0, u1), (v0, v1) = receiver.uv_extent()
    u_period_mm = getattr(receiver, "u_period_mm", None)
    k = mask_nodes
    kk = k * k
    axis_nodes = np.linspace(-support, support, k)
    au, av = np.meshgrid(axis_nodes, axis_nodes)  # rows = second angular axis
    w_nodes = kernel.value(np.hypot(au, av).ravel())  # (k²,) kernel weight per node
    w_sum = w_nodes.sum()
    d_in_nodes = -s[:, None] + au.ravel()[None, :] * e1[:, None] + av.ravel()[None, :] * e2[:, None]
    d_in_nodes /= np.linalg.norm(d_in_nodes, axis=0, keepdims=True)  # (3, k²)

    # A sample cannot be clipped if its whole angular footprint provably
    # misses every boundary that can exist with no occluders and no shadow
    # body: the secondary aperture and the receiver window. Proven, not
    # measured, its transmitted fraction is exactly 1.0 and the mask_nodes²
    # probe below is skipped for it entirely.
    can_skip = np.zeros(m, dtype=bool)
    if not occluders and shadow_body is None and not DISABLE_TRANSMISSION_SKIP:
        skip_support = _effective_support_rad(kernel)
        with np.errstate(invalid="ignore"):
            reach = _reach_mm(smax_all, hess_all, full_stencil, skip_support)
            uv0 = uv[:, 0, :]
            margin = reach * WINDOW_SAFETY_FACTOR + WINDOW_MARGIN_FLOOR_MM
            v_clears = (uv0[1] - margin >= v0) & (uv0[1] + margin <= v1)
            # A receiver that closes on itself has no edge in u (the flux
            # grid wraps there too, see `wrap_u` below) -- u0/u1 are a chart
            # cut, not a wall, so only v can clip.
            u_clears = (
                True if u_period_mm else (uv0[0] - margin >= u0) & (uv0[0] + margin <= u1)
            )
            within_window = chief_ok & can_jac & u_clears & v_clears
        if isinstance(secondary, NoSecondary):
            can_skip = within_window
        elif isinstance(secondary, (AxiconSecondary, CassegrainSecondary)) and np.any(within_window):
            cand = np.flatnonzero(within_window)
            ap_margin = max(
                SECONDARY_MARGIN_FRAC * secondary.aperture_radius_mm, SECONDARY_MARGIN_FLOOR_MM
            )
            ring_ok = _secondary_ring_clears(
                pts[:, cand],
                normal[:, cand],
                s,
                e1,
                e2,
                skip_support,
                secondary,
                ap_margin,
                RING_PROBES,
            )
            can_skip[cand] = ring_ok
        # Any other secondary (a pyramid's flat facets, or a future shape)
        # has no cheap conservative rim bound derived here, so it keeps the
        # full raster below.

    counters["transmission_skipped"] = int(can_skip.sum())
    need = ~can_skip
    frac = np.ones(m)
    node_ok = np.ones((m, kk), dtype=bool)
    uv_nodes = np.full((2, m, kk), np.nan)
    if want_sec:
        # Default 1.0 matches the skip branch's own guarantee: `can_skip`
        # only fires with no occluders and no shadow body (see above), so a
        # skipped sample has nothing shading/blocking it on the way to the
        # secondary either.
        frac_secondary = np.ones(m)

    if np.any(need):
        idxn = np.flatnonzero(need)
        pts_n = pts[:, idxn]
        normal_n = normal[:, idxn]
        mn = idxn.size

        node_ok_n = np.ones((mn, kk), dtype=bool)
        pts_t = pts_n.T
        if occluders or shadow_body is not None:
            for j in range(kk):
                toward_sun_j = -d_in_nodes[:, j]
                if occluders:
                    node_ok_n[:, j] &= ~_blocked_mask(pts_t, toward_sun_j, occluders)
                if shadow_body is not None:
                    node_ok_n[:, j] &= ~shadow_body.occludes(pts_t, toward_sun_j)

        # Reflect every node direction at every needed sample's normal and
        # push the whole bundle through the optical chain at once.
        # Sample-major layout: ray index i*k² + j is node j of sample i.
        dots = normal_n.T @ d_in_nodes  # (mn, k²)
        d_out_nodes = d_in_nodes[:, None, :] - 2.0 * dots[None, :, :] * normal_n[:, :, None]
        d_out_flat = d_out_nodes.reshape(3, mn * kk)
        p_nodes = np.repeat(pts_n, kk, axis=1)  # (3, mn*k²)
        if occluders:
            blocked_out = _blocked_mask(p_nodes.T, d_out_flat.T, occluders).reshape(mn, kk)
            node_ok_n &= ~blocked_out
        if want_sec:
            # Shading+blocking-only mask, BEFORE the receiver-window/
            # secondary-aperture filters below are ANDed in -- a free
            # byproduct captured one step earlier than the receiver deposit,
            # used as the secondary deposit's own transmitted fraction (see
            # `frac_secondary` below). A plain `.copy()`: nothing past this
            # point mutates `node_ok_n` in place other than `&=`, which
            # rebinds rather than mutating the array this aliases.
            sec_mask_n = node_ok_n.copy()
        pre_n, d_n, on_n = secondary.redirect(p_nodes, d_out_flat.copy(), {})
        if want_sec:
            # Every node ray's own secondary hit point (mm, world), scattered
            # back to sample-major (3, mn, k²) the same way `pre` was above --
            # needed for the node-fallback deposit when a sample's CHIEF ray
            # misses the secondary rim but some of its nodes still land on it.
            sec_pos_flat = np.full((3, mn * kk), np.nan)
            sec_pos_flat[:, np.flatnonzero(on_n)] = pre_n
            sec_pos_nodes_n = sec_pos_flat.reshape(3, mn, kk)
            sec_ok_n = sec_mask_n & on_n.reshape(mn, kk)
        hit_n, uv_n = receiver.intersect(pre_n, d_n)
        pass_out = np.zeros(mn * kk, dtype=bool)
        uv_nodes_n = np.full((2, mn * kk), np.nan)
        surv = np.flatnonzero(on_n)[hit_n]
        # `u` on a receiver that closes on itself is periodic, so a landing
        # point just past the seam is inside the window, not outside it --
        # wrap before comparing or a spot straddling the cut is thrown away.
        u_test = (
            uv_n[0] if not u_period_mm else u0 + np.mod(uv_n[0] - u0, u_period_mm)
        )
        in_ext = (u_test >= u0) & (u_test <= u1) & (uv_n[1] >= v0) & (uv_n[1] <= v1)
        pass_out[surv[in_ext]] = True
        uv_nodes_n[:, surv] = uv_n
        node_ok_n &= pass_out.reshape(mn, kk)
        uv_nodes_n = uv_nodes_n.reshape(2, mn, kk)

        frac[idxn] = (node_ok_n @ w_nodes) / w_sum  # kernel-weighted transmitted fraction
        node_ok[idxn] = node_ok_n
        uv_nodes[:, idxn] = uv_nodes_n
        if want_sec:
            frac_secondary[idxn] = (sec_mask_n @ w_nodes) / w_sum

    # --- classify and deposit --------------------------------------------
    cos_aoi = np.abs(normal.T @ s)  # incoming is -s; |normal . s| is cos(aoi)
    weights = STANDARD_IRRADIANCE_W_MM2 * area_w * cos_aoi

    # A receiver that closes on itself has no edge in u: the flux grid wraps.
    wrap_u = bool(u_period_mm)

    n_u, n_v = flux_grid
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    bin_area_mm2 = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])

    # `deposit_method="bspline"` accumulates onto a coarse control grid
    # instead of the fine flux grid, upsampling once at the end -- see
    # bspline_deposit.py. `accum_*` alias the fine grid unchanged when
    # deposit_method="binned" (the default and fast_accurate/MC's only
    # option), so that path is untouched, bit-for-bit, by this branch.
    use_bspline = deposit_method == "bspline"
    if use_bspline:
        if control_grid is None:
            # Proportional to flux_grid, not a fixed tuple -- see docstring:
            # a curved receiver's adaptive (wide) flux_grid needs a
            # proportionally wider control grid to keep the same
            # coarsening factor the prototype benchmarked.
            n_cu = max(CONTROL_GRID_MIN, round(n_u / CONTROL_GRID_COARSEN))
            n_cv = max(CONTROL_GRID_MIN, round(n_v / CONTROL_GRID_COARSEN))
            control_grid = (n_cu, n_cv)
        accum_u_edges, accum_v_edges = control_grid_edges(u_edges, v_edges, control_grid)
    else:
        accum_u_edges, accum_v_edges = u_edges, v_edges
    accum_n_u = accum_u_edges.size - 1
    accum_n_v = accum_v_edges.size - 1
    accum_du = accum_u_edges[1] - accum_u_edges[0]
    accum_dv = accum_v_edges[1] - accum_v_edges[0]
    accum_bin_area_mm2 = accum_du * accum_dv
    out = np.zeros((accum_n_v, accum_n_u))

    n_valid = n_masked = n_blocked = n_node_fallback = 0
    for idx in range(m):
        if frac[idx] < 1.0e-6:
            n_blocked += 1
            continue
        if chief_ok[idx] and can_jac[idx]:
            full_pass = frac[idx] > 1.0 - 1.0e-9
            hess_i = None
            if hess_all is not None and full_stencil[idx]:
                hess_i = hess_all[idx]
            deposit(
                out,
                accum_u_edges,
                accum_v_edges,
                uv[:, 0, idx],
                jac_all[idx],
                float(weights[idx]),
                kernel,
                hess=hess_i,
                mask=None if full_pass else node_ok[idx].astype(float).reshape(k, k),
                wrap_u=wrap_u,
                jac_smax=float(smax_all[idx]),
            )
            if full_pass:
                n_valid += 1
            else:
                n_masked += 1
        else:
            # The chief ray (or a whole stencil axis) is lost — usually a
            # rim-straddling sample whose kernel centre misses the aperture
            # while part of its sun cone still passes. Deposit that passing
            # mass directly at the surviving nodes' landing points,
            # kernel-weighted. Locally granular, but these slivers carry
            # little power and would otherwise be dropped entirely.
            ok_j = node_ok[idx]
            share = weights[idx] * w_nodes[ok_j] / w_sum / accum_bin_area_mm2
            iu_raw = (uv_nodes[0, idx, ok_j] - accum_u_edges[0]) // accum_du
            iv = np.clip(
                (uv_nodes[1, idx, ok_j] - accum_v_edges[0]) // accum_dv, 0, accum_n_v - 1
            ).astype(np.intp)
            if wrap_u:
                # Periodic accumulation grid: node landing points are
                # azimuth-unwrapped for stencil continuity, so a node's ``u``
                # can sit past the chart edge and its column continues at 0,
                # matching `deposit`'s own wrap_u handling above. Clipping
                # instead piles that mass onto the edge column. On the coarse
                # control grid of the bspline path that was a measured +3.7%
                # cylinder-seam peak error (coeff-prototype REPORT.md SS3.4).
                # On the binned path the reachable misplacement is kernel-tail
                # negligible (<~1e-10 of peak, measured): a chief ray near the
                # seam always INTERSECTS a surface of revolution, so fallback
                # only fires at tangent-miss limbs where the out-of-chart
                # nodes carry only the kernel's outermost weights. Wrapped on
                # both paths anyway -- same semantics, no reachable wrongness.
                iu = (iu_raw.astype(np.intp)) % accum_n_u
            else:
                iu = np.clip(iu_raw, 0, accum_n_u - 1).astype(np.intp)
            np.add.at(out, (iv, iu), share)
            n_node_fallback += 1

    counters["valid"] = n_valid
    counters["masked"] = n_masked
    counters["blocked"] = n_blocked
    counters["node_fallback"] = n_node_fallback
    counters["unresolved"] = 0  # retained for counter-invariant compatibility

    if use_bspline:
        # Upsample the coarse control grid onto the fine flux grid, once,
        # via the fixed cubic-B-spline matrices -- everything downstream
        # (power_w, flux, true-area correction) reads `out` on the fine grid
        # exactly as the binned path leaves it, so no other code path below
        # needs to know which deposit method ran.
        out = evaluate_bspline(out, accum_u_edges, accum_v_edges, u_edges, v_edges, wrap_u)

    secondary_extra: dict = {}
    if want_sec:
        (su0, su1), (sv0, sv1) = secondary_uv_extent(secondary)
        sn_u, sn_v = secondary_flux_grid
        su_edges = np.linspace(su0, su1, sn_u + 1)
        sv_edges = np.linspace(sv0, sv1, sn_v + 1)
        su_step = su_edges[1] - su_edges[0]
        sv_step = sv_edges[1] - sv_edges[0]
        sec_bin_area_mm2 = su_step * sv_step
        sec_out = np.zeros((sn_v, sn_u))

        # Chief-point deposit: every sample whose CHIEF ray reached the
        # secondary deposits its full (shading/blocking-discounted) weight
        # at that one point -- the coarse fidelity this backend documents.
        chief_idx = np.flatnonzero(chief_on_sec)
        if chief_idx.size:
            uv_c = secondary_uv(secondary, chief_sec_pts[:, chief_idx])
            w_c = weights[chief_idx] * frac_secondary[chief_idx]
            iu = np.clip((uv_c[0] - su0) // su_step, 0, sn_u - 1).astype(np.intp)
            iv = np.clip((uv_c[1] - sv0) // sv_step, 0, sn_v - 1).astype(np.intp)
            np.add.at(sec_out, (iv, iu), w_c / sec_bin_area_mm2)

        # Node fallback: a sample whose chief ray missed the secondary rim
        # (chief_on_sec False) can still have part of its kernel land on it
        # -- deposit that surviving mass at each such node's own hit point,
        # mirroring the receiver deposit's node_fallback branch above. By
        # the same invariant `_secondary_ring_clears`'s skip test relies on
        # (a skipped sample's chief ray always reaches the secondary), every
        # chief-miss sample was in `need`, so `sec_ok_n`/`sec_pos_nodes_n`
        # exist whenever this loop has work to do.
        fallback_idx = np.flatnonzero(~chief_on_sec)
        if fallback_idx.size:
            pos_in_idxn = -np.ones(m, dtype=np.intp)
            pos_in_idxn[idxn] = np.arange(idxn.size)
            for idx in fallback_idx:
                li = pos_in_idxn[idx]
                if li < 0:
                    continue  # pragma: no cover - see invariant note above
                ok_j = sec_ok_n[li]
                if not np.any(ok_j):
                    continue
                uv_j = secondary_uv(secondary, sec_pos_nodes_n[:, li, ok_j])
                share = weights[idx] * w_nodes[ok_j] / w_sum / sec_bin_area_mm2
                iu = np.clip((uv_j[0] - su0) // su_step, 0, sn_u - 1).astype(np.intp)
                iv = np.clip((uv_j[1] - sv0) // sv_step, 0, sn_v - 1).astype(np.intp)
                np.add.at(sec_out, (iv.astype(np.intp), iu.astype(np.intp)), share)

        secondary_power_w = float(sec_out.sum() * sec_bin_area_mm2)
        secondary_flux = sec_out * 1.0e6  # W/mm^2 -> W/m^2
        sec_true_area_m2 = secondary_bin_areas_m2(secondary, (sn_u, sn_v))
        sec_uniform_area_m2 = sec_bin_area_mm2 * 1.0e-6
        if not np.allclose(sec_true_area_m2, sec_uniform_area_m2):
            secondary_flux = secondary_flux * (sec_uniform_area_m2 / sec_true_area_m2)
        secondary_extra = {
            "secondary_flux": secondary_flux,
            "secondary_u_edges": su_edges,
            "secondary_v_edges": sv_edges,
            "secondary_power_w": secondary_power_w,
            # UI disclosure (spec §C): this backend's secondary deposit is a
            # chief-ray-point approximation, not a full footprint/Jacobian
            # spread -- exact accounting is Monte Carlo's
            # (`return_secondary_hits=True`).
            "secondary_fidelity": "coarse",
        }

    bin_area_mm2 = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])
    power_w = float(out.sum() * bin_area_mm2)
    flux = out * 1.0e6  # W/mm^2 -> W/m^2
    # `out` is power per (u, v) parameter bin. On a frustum the parameter
    # bins are uniform but the surface rows they map to are not (area scales
    # with r(v)/r_mean), so the physical W/m^2 divides bin power by the TRUE
    # row area — the same bin_areas_m2 the MC backend, _mean_flux_kw_m2 and
    # the FEA CSV already use. Flat/cylinder areas are uniform: untouched.
    true_area_m2 = receiver.bin_areas_m2((n_u, n_v))
    uniform_area_m2 = bin_area_mm2 * 1.0e-6
    if not np.allclose(true_area_m2, uniform_area_m2):
        flux *= uniform_area_m2 / true_area_m2
    return {
        "flux": flux,
        "u_edges": u_edges,
        "v_edges": v_edges,
        "power_w": power_w,
        "incident_power_w": float(weights.sum()),
        "counters": counters,
        "chief_uv": uv[:, 0],
        "jacobians": jac_all,
        **secondary_extra,
    }
