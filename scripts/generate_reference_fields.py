"""Generate the four §P reference-field position CSVs.

Run once (``python scripts/generate_reference_fields.py``) to (re)write the
four CSVs under ``src/heliostat/web/static/data/`` that
``heliostat.web.builtin_library.BUILTIN_PROJECTS`` loads at import time:
``field_gemasolar.csv``, ``field_ps10.csv``, ``field_crescent_dunes.csv``,
``field_hami.csv``. Nothing at runtime regenerates these -- like
``field_645.csv`` (the manuscript's own packaged field), they are committed
data, and this script is the record of how they were produced, not a
service the app calls.

**These are RECONSTRUCTIONS, not real plant layouts.** Gemasolar, PS10, and
Crescent Dunes's exact as-built heliostat coordinates are proprietary and
unpublished (see ``docs/ui-spec-v0.2.md`` §P and the research this rider
built from,
``scratchpad/reference_fields_research.md`` in the session that wrote this
script). Every field below is generated from the *published, aggregate*
parameters that research pass found (heliostat count, mirror size, tower/
receiver height, approximate field radius) using one straightforward
radial-staggered placement rule -- not fit to any leaked or scraped
coordinate file, because none exists publicly for any of these four plants.
Confidence labels (DOCUMENTED / WIDELY REPORTED / INFERRED) on every number
below are carried verbatim from that research pass; anything not labelled
DOCUMENTED in a field's block comment is this script's own choice, not a
published fact, and is disclosed as such in the matching
``heliostat.web.builtin_library.BUILTIN_PROJECT_PROVENANCE`` entry that
ships with the project.

**The placement rule** (``radial_stagger_field`` below) is NOT any of the
four plants' real ring geometry -- none of that is published for any of
them. It is one algorithm, applied identically to all four (and to PS10's
north sector), so the same disclosed approximation is visible everywhere it
appears rather than four different unstated fits:

1. Start at an inner stand-off radius ``r0_m`` (this script's own choice --
   see each field's block below for the reasoning), one ring at a time.
2. Each ring's heliostat count is chosen so neighbouring heliostats on that
   ring are spaced roughly ``az_pitch_m`` apart (arc length) -- by default
   1.3x the heliostat's own width, a generic, disclosed packing-density
   choice, not a per-plant number.
3. Consecutive rings alternate a half-pitch azimuthal stagger (the
   "radial-staggered" pattern this app's own default field and
   ``RadialStaggeredLayout`` in ``heliostat.web.app`` already use).
4. The radial step out to the next ring starts at ``radial_pitch_m``
   (default 1.3x the heliostat's own height) and grows a further 5% per
   ring index -- a deliberately gentle, generic growth rule chosen only to
   avoid a perfectly uniform grid; it reproduces neither plant's real
   radial spacing, which is unpublished for all four.
5. Stops once ``n_target`` heliostats have been placed; the last ring is
   truncated to hit the count exactly.

A field's ``az_min_deg``/``az_max_deg`` (compass convention: 0 deg = north,
clockwise) restricts placement to a sector -- used for PS10's north field
only. Ring counts are computed from the sector's own arc length, not
generated full-circle and filtered afterwards, so the azimuthal pitch stays
correct at the sector edges. This intentionally does NOT import
``heliostat.field_layouts.wedge_filter``: that helper uses the *math*
angle convention (0 deg = +x, counter-clockwise -- see its own docstring's
explicit warning), not the compass convention this script and
``HeliostatField.azimuth_deg`` use everywhere else, and mixing the two
conventions in one script is exactly the kind of silent-bug risk that
module's own docstring flags. ``heliostat.field_layouts.ring_filter`` (r_min/
r_max only, convention-agnostic) would be usable here but adds nothing this
script's own radial loop bounds don't already give it directly, so it is
not used either -- kept in mind as the natural helper if a future session
wants a filter-composition style instead of a generator loop.

Every generated field is written with :func:`heliostat.field_layouts.write_field_csv`
(metre-unit ``x (m)``/``y (m)`` columns, one of
:func:`heliostat.field.load_field`'s recognised header aliases), so it
loads back byte-for-byte the same way ``field_645.csv`` does.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from heliostat.field import HeliostatField
from heliostat.field_layouts import write_field_csv

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "heliostat" / "web" / "static" / "data"

#: Ring-to-ring radial growth: each ring's own radial step is this much
#: larger than the previous ring's, a fraction per ring index -- see the
#: module docstring's point 4. Same constant for all four fields.
RADIAL_GROWTH_PER_RING = 0.05

#: Default azimuthal/radial packing pitch, as a multiple of the heliostat's
#: own width/height -- see the module docstring's point 2/4. Same constant
#: for all four fields; not a per-plant published number.
PACKING_FACTOR = 1.3


def radial_stagger_field(
    n_target: int,
    r0_m: float,
    heliostat_width_m: float,
    heliostat_height_m: float,
    az_min_deg: float | None = None,
    az_max_deg: float | None = None,
    packing_factor: float = PACKING_FACTOR,
    growth_per_ring: float = RADIAL_GROWTH_PER_RING,
) -> np.ndarray:
    """``(n_target, 2)`` array of (x, y) metres, compass convention
    (0 deg = +y/north, clockwise: ``x = r*sin(az)``, ``y = r*cos(az)``,
    matching :attr:`heliostat.field.HeliostatField.azimuth_deg`).

    See the module docstring for the placement rule. ``az_min_deg``/
    ``az_max_deg`` (compass degrees) restrict every ring to a sector,
    wrapping through 0 deg if ``az_min_deg > az_max_deg``; ``None``/``None``
    (the default) is full surround.
    """
    az_pitch_m = packing_factor * heliostat_width_m
    radial_pitch0_m = packing_factor * heliostat_height_m
    sector = az_min_deg is not None
    span_deg = 360.0 if not sector else ((az_max_deg - az_min_deg) % 360.0 or 360.0)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total = 0
    ring = 0
    r = r0_m
    while total < n_target:
        circumference_m = 2.0 * math.pi * r * (span_deg / 360.0)
        n_ring = max(1, round(circumference_m / az_pitch_m))
        n_ring = min(n_ring, n_target - total)
        pitch_deg = span_deg / n_ring
        stagger_deg = pitch_deg / 2.0 if ring % 2 == 0 else 0.0
        base_deg = stagger_deg if not sector else az_min_deg + stagger_deg
        az_deg = base_deg + pitch_deg * np.arange(n_ring)
        az_rad = np.radians(az_deg)
        xs.append(r * np.sin(az_rad))
        ys.append(r * np.cos(az_rad))
        total += n_ring
        ring += 1
        r += radial_pitch0_m * (1.0 + growth_per_ring * (ring - 1))

    x = np.concatenate(xs)[:n_target]
    y = np.concatenate(ys)[:n_target]
    return np.column_stack((x, y))


def _write(name: str, xy_m: np.ndarray, source: str) -> None:
    field = HeliostatField(
        x_mm=xy_m[:, 0] * 1000.0,
        y_mm=xy_m[:, 1] * 1000.0,
        ids=np.arange(xy_m.shape[0], dtype=int),
        source=source,
    )
    path = DATA_DIR / name
    write_field_csv(field, path)
    r_m = np.hypot(xy_m[:, 0], xy_m[:, 1])
    print(f"{name}: {len(field)} heliostats, radius {r_m.min():.1f}-{r_m.max():.1f} m -> {path}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- Gemasolar (Torresol Energy, Fuentes de Andalucia, Spain) -----------
    # Heliostat count 2650, mirror 11.5 x 10.4 m: DOCUMENTED (HelioCon,
    # citing the Sener spec sheet; see BUILTIN_PROJECT_PROVENANCE for the
    # full citation). r0_m=120 (inner stand-off) is this script's own
    # choice -- not published -- picked as a plausible fraction (~0.86x) of
    # the 140 m tower height. Full surround: real Gemasolar is denser to the
    # north than south (WIDELY REPORTED), which this circularly-symmetric
    # algorithm cannot reproduce -- disclosed in the provenance text, not
    # silently smoothed over.
    xy = radial_stagger_field(
        n_target=2650, r0_m=120.0, heliostat_width_m=11.5, heliostat_height_m=10.4
    )
    _write(
        "field_gemasolar.csv",
        xy,
        "radial_stagger_field(n_target=2650, r0_m=120.0, w=11.5, h=10.4) -- "
        "reconstruction, see docs/ui-spec-v0.2.md §P",
    )

    # -- PS10 (Abengoa/Solucar, Sanlucar la Mayor, Spain) --------------------
    # Heliostat count 624: DOCUMENTED (Wikipedia, Modern Power Systems).
    # Mirror dims not published -- INFERRED as a ~10.977 m square from the
    # documented ~120.5 m^2 unit area. North sector only (DOCUMENTED --
    # "grouped on the north side" per Modern Power Systems); half-span
    # +-60 deg reuses this app's own (as yet unbuilt) §I sector-field
    # default rather than inventing a new number -- PS10's real angular
    # span is not published. r0_m=60 is this script's own choice (~0.52x
    # the 115 m tower height), not published.
    ps10_side_m = math.sqrt(120.5)
    xy = radial_stagger_field(
        n_target=624,
        r0_m=60.0,
        heliostat_width_m=ps10_side_m,
        heliostat_height_m=ps10_side_m,
        az_min_deg=-60.0,
        az_max_deg=60.0,
    )
    _write(
        "field_ps10.csv",
        xy,
        "radial_stagger_field(n_target=624, r0_m=60.0, w=h=10.977, "
        "az_min_deg=-60, az_max_deg=60) -- reconstruction, see docs/ui-spec-v0.2.md §P",
    )

    # -- Crescent Dunes (SolarReserve, Tonopah, Nevada) ----------------------
    # Heliostat count 10347, mirror 10.8 x 10.8 m: DOCUMENTED (HelioCon).
    # r0_m=160 (~0.82x the 195 m tower height) is this script's own choice,
    # not published. Full surround (WIDELY REPORTED).
    xy = radial_stagger_field(
        n_target=10347, r0_m=160.0, heliostat_width_m=10.8, heliostat_height_m=10.8
    )
    _write(
        "field_crescent_dunes.csv",
        xy,
        "radial_stagger_field(n_target=10347, r0_m=160.0, w=h=10.8) -- "
        "reconstruction, see docs/ui-spec-v0.2.md §P",
    )

    # -- Hami (Stellio-based field; CEEC/sbp, Xinjiang, China) ---------------
    # Heliostat count 14500: sbp's own official number, DOCUMENTED; other
    # sources report 14000 (HelioCon) or 15000 (SolarPACES, PowerMag) for
    # the same plant -- disclosed as a disputed range in the provenance
    # text, not silently picked. Pentagon heliostat footprint ~8.94 m
    # circumdiameter -- matching BUILTIN_PROJECTS's actual regular-pentagon
    # custom-polygon design (circumradius 4469.65 mm, area-matched to the
    # documented ~47.5 m^2), not the ~7.9 m figure a separate, weaker
    # HelioCon estimate gives (see BUILTIN_PROJECT_PROVENANCE's caveat on
    # that inconsistency) -- stands in for width/height in the packing
    # pitch below. r0_m=180 (~0.82x the 220 m tower height, itself a
    # single-source figure) is this script's own choice, not published.
    # Full surround (WIDELY REPORTED, from the AIP paper's abstract).
    hami_footprint_m = 8.94
    xy = radial_stagger_field(
        n_target=14500,
        r0_m=180.0,
        heliostat_width_m=hami_footprint_m,
        heliostat_height_m=hami_footprint_m,
    )
    _write(
        "field_hami.csv",
        xy,
        "radial_stagger_field(n_target=14500, r0_m=180.0, w=h=8.94) -- "
        "reconstruction, see docs/ui-spec-v0.2.md §P",
    )


if __name__ == "__main__":
    main()
