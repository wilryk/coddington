"""On-disk store for trace results.

Layout under a run's root directory::

    manifest.json                run metadata + quantisation scale
    summary.csv                  one row per (timestep, heliostat)
    raw/<key>_rays.npy           int16 (N, 2)  receiver x/y, all heliostats concatenated
    raw/<key>_index.npy          int64 (H, 3)  [heliostat_id, start, count]
    flux/<key>.npy               uint32 (H, G, G)  per-heliostat bin counts

Design notes
------------
**Counts are stored, never scaled flux.** Watts-per-ray, mirror throughput,
and DNI are all applied at read time by :func:`flux_scale`. Every one of
those can therefore be revised without re-tracing.

**Raw rays are the source of truth; flux maps are a cache.** They are written
alongside the trace because binning during the trace is nearly free, but they
are fully reconstructible from the raw rays via :meth:`RunStore.rebin`.

**int16 quantisation** over the +/-window_mm receiver window gives fine
sub-millimetre resolution at half the size of float32 -- far finer than a
typical flux bin, and irrelevant next to Monte-Carlo noise.

CSV (not Parquet) for the summary: a run's summary is trivially small, it
needs no extra dependency, and it opens in a spreadsheet.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INT16_MAX = 32767
SUMMARY_NAME = "summary.csv"
MANIFEST_NAME = "manifest.json"

#: flux_kind values :func:`flux_scale` understands. ``"ray_counts"`` is a
#: per-bin count of quantised rays, scaled by watts-per-ray; ``"analytic"``
#: is a per-bin power already in the trace's native units (e.g. an
#: analytic/kernel-deposit backend), scaled only by throughput and DNI.
FLUX_KINDS = ("ray_counts", "analytic")


@dataclass
class TimestepResult:
    """Everything one traced timestep contributes to the store."""

    key: str
    date: str
    hour: float
    solar_az_deg: float
    solar_el_deg: float
    heliostat_ids: np.ndarray
    rays_emitted: int
    counts: np.ndarray  # uint32 (H, G, G)
    rays: np.ndarray | None  # int16 (N, 2), or None when raw retention is off
    index: np.ndarray | None  # int64 (H, 3)
    rows: pd.DataFrame  # per-heliostat summary rows


def flux_scale(
    cfg, rays_emitted: int, dni_w_m2: float = 1000.0, flux_kind: str = "ray_counts"
) -> float:
    """The single choke point for turning a stored bin count into watts.

    Multiply a bin count by this and divide by bin area to get W/m^2.

    ``flux_kind`` selects the formula:

    ``"ray_counts"``
        A bin holds a count of quantised rays. Scaled by watts-per-ray
        (``cfg.source.watts_per_ray(rays_emitted)``), mirror throughput, and
        DNI. This is the Monte Carlo path and the historical default.

    ``"analytic"``
        A bin already holds power in the trace's native (DNI=1000, no
        throughput) units -- e.g. a kernel-deposit backend that writes power
        directly rather than ray counts. Scaled only by throughput and DNI;
        there is no ray budget to divide by.

    Every reader of a stored run should call this (or :func:`scale_factor`
    for the common ray-count case) rather than reimplementing the formula,
    so a change to what "watts per stored unit" means only has to happen
    once.
    """
    if flux_kind == "ray_counts":
        return cfg.source.watts_per_ray(rays_emitted) * cfg.optics.throughput * (dni_w_m2 / 1000.0)
    if flux_kind == "analytic":
        return cfg.optics.throughput * (dni_w_m2 / 1000.0)
    raise ValueError(f"unknown flux_kind {flux_kind!r}; use one of {FLUX_KINDS}")


def scale_factor(cfg, rays_emitted: int, dni_w_m2: float = 1000.0) -> float:
    """Watts per landed ray, including throughput and DNI.

    The ``flux_kind="ray_counts"`` case of :func:`flux_scale`, kept as its
    own name because it is the common case and the one every other module
    (``metrics``, ``energy``) was written against. This is the single owner
    of the formula -- other modules import it from here rather than keeping
    their own copy.
    """
    return flux_scale(cfg, rays_emitted, dni_w_m2, flux_kind="ray_counts")


class RunStore:
    """Reader/writer for a trace output directory."""

    def __init__(self, root, cfg=None, mode: str = "r"):
        self.root = Path(root)
        self.cfg = cfg
        self.mode = mode
        self.raw_dir = self.root / "raw"
        self.flux_dir = self.root / "flux"
        if mode == "w":
            for d in (self.root, self.raw_dir, self.flux_dir):
                d.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(f"No store at {self.root}")
        self._manifest: dict | None = None

    # -- manifest ---------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            path = self.root / MANIFEST_NAME
            self._manifest = json.loads(path.read_text()) if path.exists() else {}
        return self._manifest

    def write_manifest(
        self,
        cfg,
        *,
        receiver=None,
        design=None,
        flux_kind: str = "ray_counts",
        extra: dict | None = None,
    ) -> None:
        """Write ``manifest.json``.

        ``cfg`` is duck-typed; the fields this method reads are exactly:
        ``cfg.receiver.window_mm``, ``cfg.receiver.grid_size``,
        ``cfg.trace.rays_per_heliostat``, ``cfg.source.power_w``,
        ``cfg.optics.throughput``, ``cfg.storage.raw_rays`` -- the same
        small surface :mod:`heliostat.metrics` and the trace modules read.

        ``receiver`` (a :class:`heliostat.geometry.receiver.Receiver`) and
        ``design`` (a :class:`heliostat.geometry.design.HeliostatDesign`),
        when given, are recorded via their own ``to_manifest``/``to_dict``
        and can be rebuilt with :meth:`receiver_from_manifest` and
        :meth:`design_from_manifest`. Both are optional and absent by
        default -- a manifest with neither key describes a run the same way
        it always has.

        ``flux_kind`` is recorded so a reader knows which :func:`flux_scale`
        formula the stored counts need; see there for what each value means.

        Anything else a caller wants recorded (site, geometry, run options,
        provenance...) goes through ``extra``, which is applied last and can
        override any of the above -- this method does not prescribe a run's
        full metadata shape, only the fields it itself depends on to read
        counts back as watts.
        """
        payload = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "quantisation_scale_mm": cfg.receiver.window_mm / INT16_MAX,
            "receiver_window_mm": cfg.receiver.window_mm,
            "grid_size": cfg.receiver.grid_size,
            "rays_per_heliostat": cfg.trace.rays_per_heliostat,
            "source_power_w": cfg.source.power_w,
            "throughput": cfg.optics.throughput,
            "flux_kind": flux_kind,
            "raw_rays": cfg.storage.raw_rays,
        }
        if receiver is not None:
            payload["receiver"] = receiver.to_manifest()
        if design is not None:
            payload["design"] = design.to_dict()
        payload.update(extra or {})
        self._manifest = payload
        (self.root / MANIFEST_NAME).write_text(json.dumps(payload, indent=2))

    @property
    def quant_scale(self) -> float:
        return float(self.manifest.get("quantisation_scale_mm", 1.0))

    @property
    def flux_kind(self) -> str:
        return str(self.manifest.get("flux_kind", "ray_counts"))

    def receiver_from_manifest(self):
        """Rebuild the run's :class:`~heliostat.geometry.receiver.Receiver`.

        Returns ``None`` when the manifest carries no ``"receiver"`` key --
        older or minimal manifests describe a run this way, and callers must
        not have to distinguish that from an error.
        """
        entry = self.manifest.get("receiver")
        if entry is None:
            return None
        from .geometry.receiver import Receiver

        return Receiver.from_manifest(entry)

    def design_from_manifest(self):
        """Rebuild the run's :class:`~heliostat.geometry.design.HeliostatDesign`.

        Returns ``None`` when the manifest carries no ``"design"`` key.
        """
        entry = self.manifest.get("design")
        if entry is None:
            return None
        from .geometry.design import HeliostatDesign

        return HeliostatDesign.from_dict(entry)

    # -- quantisation -----------------------------------------------------
    @staticmethod
    def inside_window(xy_mm: np.ndarray, window_mm: float) -> np.ndarray:
        """Mask of rays within the storable receiver window."""
        return (np.abs(xy_mm[:, 0]) <= window_mm) & (np.abs(xy_mm[:, 1]) <= window_mm)

    @staticmethod
    def quantise(xy_mm: np.ndarray, window_mm: float) -> np.ndarray:
        """Float mm -> int16.

        Rays must already be inside the window -- use :meth:`inside_window` to
        filter first. Clipping here instead would pile out-of-window rays onto
        the boundary, inventing a hot ring at the receiver edge and making the
        raw store disagree with the binned counts.
        """
        scaled = np.clip(xy_mm / window_mm, -1.0, 1.0) * INT16_MAX
        return np.rint(scaled).astype(np.int16)

    def dequantise(self, raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32) * np.float32(self.quant_scale)

    # -- writing ----------------------------------------------------------
    def write_timestep(self, result: TimestepResult) -> None:
        np.save(self.flux_dir / f"{result.key}.npy", result.counts)
        if result.rays is not None and result.index is not None:
            np.save(self.raw_dir / f"{result.key}_rays.npy", result.rays)
            np.save(self.raw_dir / f"{result.key}_index.npy", result.index)
        self.append_summary(result.rows)

    def append_summary(self, rows: pd.DataFrame) -> None:
        path = self.root / SUMMARY_NAME
        rows.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)

    # -- reading ----------------------------------------------------------
    def timestep_keys(self) -> list[str]:
        return sorted(p.stem for p in self.flux_dir.glob("*.npy"))

    def has_timestep(self, key: str) -> bool:
        """Used to make a trace resumable after an interruption."""
        return (self.flux_dir / f"{key}.npy").exists()

    def read_counts(self, key: str, mmap: bool = True) -> np.ndarray:
        """Per-heliostat bin counts, shape (H, G, G) in the manifest's
        ``heliostat_ids`` order."""
        return np.load(self.flux_dir / f"{key}.npy", mmap_mode="r" if mmap else None)

    def read_index(self, key: str) -> np.ndarray:
        """Raw-ray index [heliostat_id, start, count]."""
        return np.load(self.raw_dir / f"{key}_index.npy")

    def read_rays(self, key: str, heliostat_id: int | None = None) -> np.ndarray:
        """Receiver x/y in mm. One heliostat, or all of them concatenated.

        Memory-maps the file and slices, so reading one heliostat out of a
        large timestep does not read the whole thing.
        """
        rays_path = self.raw_dir / f"{key}_rays.npy"
        if not rays_path.exists():
            raise FileNotFoundError(
                f"No raw rays for {key} (storage.raw_rays was "
                f"{self.manifest.get('raw_rays')!r} for this run)"
            )
        raw = np.load(rays_path, mmap_mode="r")
        if heliostat_id is None:
            return self.dequantise(np.asarray(raw))

        index = self.read_index(key)
        match = index[index[:, 0] == heliostat_id]
        if match.size == 0:
            raise KeyError(f"heliostat {heliostat_id} not in timestep {key}")
        _, start, count = match[0]
        return self.dequantise(np.asarray(raw[start : start + count]))

    def rebin(
        self, key: str, grid_size: int, window_mm: float, heliostat_id: int | None = None
    ) -> np.ndarray:
        """Re-histogram raw rays at a different resolution or window."""
        xy = self.read_rays(key, heliostat_id)
        edges = np.linspace(-window_mm, window_mm, grid_size + 1)
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[edges, edges])
        return counts

    # -- summary ----------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """The run's summary as one row per (timestep, heliostat).

        Coincident duplicate heliostats are not filtered here: the field
        loader (:func:`heliostat.field.load_field`) already refuses to build
        a field with coincident positions, so a run store constructed from
        this package's field loader cannot contain them in the first place.
        The private predecessor of this module carried a read-time dedup for
        runs traced before that refusal existed; this port has no such
        history to paper over.
        """
        path = self.root / SUMMARY_NAME
        if not path.exists():
            raise FileNotFoundError(f"No summary at {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # -- aggregation ------------------------------------------------------
    def field_flux(
        self, key: str, cfg=None, dni_w_m2: float = 1000.0, efficiency: np.ndarray | None = None
    ) -> np.ndarray:
        """Whole-field receiver flux in W/m^2 for one timestep.

        Flux maps add linearly, so the combined field map is a weighted sum over
        per-heliostat maps -- no re-trace, and per-heliostat efficiency factors
        (shading, blocking) drop straight in as weights.
        """
        cfg = cfg or self.cfg
        counts = np.asarray(self.read_counts(key)).astype(np.float64)
        eff = None if efficiency is None else np.asarray(efficiency, float)
        if eff is not None:
            if eff.shape[0] != counts.shape[0]:
                raise ValueError(
                    f"efficiency has {eff.shape[0]} rows, flux has "
                    f"{counts.shape[0]} -- the caller must weight from the "
                    f"same heliostat ordering the run's counts use"
                )
            counts = counts * eff[:, None, None]
        total = counts.sum(axis=0)
        rays_emitted = int(self.manifest.get("rays_per_heliostat", cfg.trace.rays_per_heliostat))
        scale = flux_scale(cfg, rays_emitted, dni_w_m2, self.flux_kind)
        return total * scale / cfg.receiver.bin_area_m2

    def heliostat_flux(
        self,
        key: str,
        heliostat_row: int,
        cfg=None,
        dni_w_m2: float = 1000.0,
        efficiency: float = 1.0,
    ) -> np.ndarray:
        """Single-heliostat receiver flux in W/m^2."""
        cfg = cfg or self.cfg
        counts = np.asarray(self.read_counts(key)[heliostat_row]).astype(np.float64) * efficiency
        rays_emitted = int(self.manifest.get("rays_per_heliostat", cfg.trace.rays_per_heliostat))
        scale = flux_scale(cfg, rays_emitted, dni_w_m2, self.flux_kind)
        return counts * scale / cfg.receiver.bin_area_m2
