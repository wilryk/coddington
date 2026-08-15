"""Per-heliostat scalar metrics.

These are what turn many flux maps into a table you can sort. Computed from
the raw receiver rays where available (exact) and from binned counts
otherwise (resolution-limited but cheap).

Everything here is DNI-independent unless a DNI value is passed, matching the
convention that counts are the durable quantity and all scaling is applied at
read time.
"""

from __future__ import annotations

import numpy as np


def scale_factor(cfg, rays_emitted: int, dni_w_m2: float = 1000.0) -> float:
    """Watts per landed ray, including reflectivity and DNI.

    Multiply a bin count by this and divide by bin area to get W/m^2.
    """
    return cfg.source.watts_per_ray(rays_emitted) * cfg.optics.throughput * (dni_w_m2 / 1000.0)


def bin_centres(cfg, grid_size: int | None = None) -> np.ndarray:
    """Centre coordinate of each receiver bin along one axis, in mm.

    ``grid_size`` overrides the configured one, for a map that has been
    re-histogrammed at a different resolution -- the window is unchanged, so the
    bins simply get larger or smaller.
    """
    w = cfg.receiver.window_mm
    g = int(grid_size or cfg.receiver.grid_size)
    return np.linspace(-w, w, g + 1)[:-1] + w / g


def bin_radius(cfg, grid_size: int | None = None) -> np.ndarray:
    """Distance of each bin centre from the receiver axis, shape (G, G).

    The receiver *axis* -- not the spot centroid -- because that is what an
    aperture actually intercepts. A spot whose centroid has walked off axis
    spills, and a centroid-referenced radius would hide exactly that.
    """
    c = bin_centres(cfg, grid_size)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.hypot(xx, yy)


def radial_mask(cfg, radius_mm: float, grid_size: int | None = None) -> np.ndarray:
    """Boolean (G, G) mask of the bins inside an aperture of this radius."""
    return bin_radius(cfg, grid_size) <= float(radius_mm)


def radial_masks(cfg, radii_mm, grid_size: int | None = None) -> np.ndarray:
    """One :func:`radial_mask` per radius, stacked."""
    rr = bin_radius(cfg, grid_size)
    return np.stack([rr <= float(r) for r in np.asarray(radii_mm, float)])


def aperture_metrics(
    counts: np.ndarray,
    rays_emitted: int,
    cfg,
    radius_mm: float,
    dni_w_m2: float = 1000.0,
    efficiency: float = 1.0,
) -> dict:
    """Power delivered inside an aperture, from a binned map.

    The masked-sum analogue of :func:`spot_metrics`'s ``aperture_radius_mm``
    branch, for the case where only counts are to hand. ``spillage`` is defined
    identically -- the fraction of *landed* rays that miss the aperture -- so the
    two paths can be cross-checked against each other.
    """
    counts = np.asarray(counts, dtype=float)
    watts_per_ray = scale_factor(cfg, rays_emitted, dni_w_m2) * efficiency
    inside = counts[radial_mask(cfg, radius_mm)]
    landed = float(counts.sum())
    captured = float(inside.sum())

    return {
        "power_w": captured * watts_per_ray,
        "power_total_w": landed * watts_per_ray,
        "peak_flux_w_m2": float(inside.max() * watts_per_ray / cfg.receiver.bin_area_m2)
        if inside.size
        else 0.0,
        "spillage": float(1.0 - captured / landed) if landed else float("nan"),
    }


def encircled_energy(
    counts: np.ndarray,
    rays_emitted: int,
    cfg,
    dni_w_m2: float = 1000.0,
    efficiency: float = 1.0,
    n_radii: int = 240,
):
    """Cumulative power vs aperture radius, from a binned map.

    Returns ``(radii_mm, power_w, fraction)`` on a uniform radius grid, so two
    curves computed from different maps can be compared or differenced directly.
    Measured about the receiver axis, matching :func:`aperture_metrics`.
    """
    counts = np.asarray(counts, dtype=float)
    watts_per_ray = scale_factor(cfg, rays_emitted, dni_w_m2) * efficiency
    rr = bin_radius(cfg).ravel()

    order = np.argsort(rr)
    cum = np.concatenate(([0.0], np.cumsum(counts.ravel()[order]) * watts_per_ray))
    radii = np.linspace(0.0, cfg.receiver.window_mm, n_radii)
    # searchsorted, not interp: a bin is either inside the aperture or it is not,
    # so the curve is a right-continuous step. Interpolating between bins would
    # make this disagree with radial_mask by a fraction of a percent, and the two
    # are supposed to be the same definition.
    power = cum[np.searchsorted(rr[order], radii, side="right")]
    total = float(cum[-1])
    return radii, power, power / total if total else np.zeros_like(power)


def encircled_energy_rays(
    xy_mm: np.ndarray,
    rays_emitted: int,
    cfg,
    dni_w_m2: float = 1000.0,
    efficiency: float = 1.0,
    n_radii: int = 240,
):
    """Same curve from raw rays -- exact, with no bin quantisation."""
    watts_per_ray = scale_factor(cfg, rays_emitted, dni_w_m2) * efficiency
    radii = np.linspace(0.0, cfg.receiver.window_mm, n_radii)
    if xy_mm.shape[0] == 0:
        z = np.zeros_like(radii)
        return radii, z, z

    r = np.sort(np.hypot(xy_mm[:, 0], xy_mm[:, 1]))
    power = np.searchsorted(r, radii, side="right") * watts_per_ray
    total = float(r.size * watts_per_ray)
    return radii, power, power / total if total else np.zeros_like(power)


def encircled_energy_radii(
    xy_mm: np.ndarray, fractions=(0.5, 0.9), centre: np.ndarray | None = None
) -> dict[float, float]:
    """Radii containing the given fractions of the landed rays, in mm."""
    if xy_mm.shape[0] == 0:
        return {f: float("nan") for f in fractions}
    centre = xy_mm.mean(axis=0) if centre is None else centre
    r = np.sort(np.hypot(xy_mm[:, 0] - centre[0], xy_mm[:, 1] - centre[1]))
    n = r.size
    return {f: float(r[min(n - 1, max(0, int(np.ceil(f * n)) - 1))]) for f in fractions}


def spot_metrics(
    xy_mm: np.ndarray,
    rays_emitted: int,
    cfg,
    dni_w_m2: float = 1000.0,
    efficiency: float = 1.0,
    aperture_radius_mm: float | None = None,
) -> dict:
    """Full metric set for one heliostat at one instant, from raw rays.

    ``efficiency`` folds in shading and blocking; ``aperture_radius_mm`` defines
    what counts as spillage.
    """
    landed = int(xy_mm.shape[0])
    watts_per_ray = scale_factor(cfg, rays_emitted, dni_w_m2) * efficiency
    power_w = landed * watts_per_ray

    out = {
        "rays_emitted": int(rays_emitted),
        "rays_landed": landed,
        "transmission": landed / rays_emitted if rays_emitted else 0.0,
        "power_w": power_w,
        "shading_blocking_efficiency": float(efficiency),
    }

    if landed == 0:
        out.update(
            {
                "centroid_x_mm": float("nan"),
                "centroid_y_mm": float("nan"),
                "rms_radius_mm": float("nan"),
                "r50_mm": float("nan"),
                "r90_mm": float("nan"),
                "peak_flux_w_m2": 0.0,
                "spillage": float("nan"),
            }
        )
        return out

    centre = xy_mm.mean(axis=0)
    r = np.hypot(xy_mm[:, 0] - centre[0], xy_mm[:, 1] - centre[1])
    radii = encircled_energy_radii(xy_mm, (0.5, 0.9), centre)

    edges = cfg.receiver.edges
    counts, _, _ = np.histogram2d(xy_mm[:, 1], xy_mm[:, 0], bins=[edges, edges])
    peak_flux = counts.max() * watts_per_ray / cfg.receiver.bin_area_m2

    out.update(
        {
            "centroid_x_mm": float(centre[0]),
            "centroid_y_mm": float(centre[1]),
            "rms_radius_mm": float(np.sqrt(np.mean(r**2))),
            "r50_mm": radii[0.5],
            "r90_mm": radii[0.9],
            "peak_flux_w_m2": float(peak_flux),
        }
    )

    if aperture_radius_mm is not None:
        inside = np.hypot(xy_mm[:, 0], xy_mm[:, 1]) <= aperture_radius_mm
        out["spillage"] = float(1.0 - inside.mean())
        out["power_in_aperture_w"] = float(inside.sum()) * watts_per_ray
    else:
        out["spillage"] = float("nan")

    return out


def map_metrics(
    counts: np.ndarray, rays_emitted: int, cfg, dni_w_m2: float = 1000.0, efficiency: float = 1.0
) -> dict:
    """Metrics from a binned map, for when raw rays were not retained."""
    counts = np.asarray(counts, dtype=float)
    watts_per_ray = scale_factor(cfg, rays_emitted, dni_w_m2) * efficiency
    landed = float(counts.sum())

    g = cfg.receiver.grid_size
    centres = np.linspace(
        -cfg.receiver.window_mm + cfg.receiver.bin_size_mm / 2,
        cfg.receiver.window_mm - cfg.receiver.bin_size_mm / 2,
        g,
    )
    X, Y = np.meshgrid(centres, centres)

    if landed == 0:
        return {
            "rays_landed": 0,
            "power_w": 0.0,
            "peak_flux_w_m2": 0.0,
            "centroid_x_mm": float("nan"),
            "centroid_y_mm": float("nan"),
        }

    cx = float((X * counts).sum() / landed)
    cy = float((Y * counts).sum() / landed)
    r = np.hypot(X - cx, Y - cy)

    order = np.argsort(r.ravel())
    cum = np.cumsum(counts.ravel()[order]) / landed
    r_sorted = r.ravel()[order]

    return {
        "rays_landed": int(landed),
        "power_w": landed * watts_per_ray,
        "peak_flux_w_m2": float(counts.max() * watts_per_ray / cfg.receiver.bin_area_m2),
        "centroid_x_mm": cx,
        "centroid_y_mm": cy,
        "rms_radius_mm": float(np.sqrt((counts * r**2).sum() / landed)),
        "r50_mm": float(r_sorted[np.searchsorted(cum, 0.5)]),
        "r90_mm": float(r_sorted[np.searchsorted(cum, 0.9)]),
    }


def rank_heliostats(summary, by: str = "power_w", ascending: bool = True):
    """Aggregate the summary table to one row per heliostat and rank it.

    ``by`` is any per-timestep column; it is summed over time for extensive
    quantities and averaged for intensive ones.
    """
    intensive = {
        "transmission",
        "shading_blocking_efficiency",
        "peak_flux_w_m2",
        "r50_mm",
        "r90_mm",
        "rms_radius_mm",
        "spillage",
        "cosine_efficiency",
    }
    agg = "mean" if by in intensive else "sum"

    grouped = (
        summary.groupby("heliostat_id")
        .agg(
            x_m=("x_m", "first"),
            y_m=("y_m", "first"),
            radius_m=("radius_m", "first"),
            n_timesteps=("hour", "count"),
            value=(by, agg),
            power_w_total=("power_w", "sum"),
            mean_transmission=("transmission", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values("value", ascending=ascending).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped = grouped.rename(columns={"value": f"{by}_{agg}"})
    return grouped
