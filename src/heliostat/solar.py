"""Solar position, sunrise/sunset, and the sweep time grid.

The core algorithm is the NOAA spreadsheet method, ported from a MATLAB
NOAA solar-position implementation (SolarPositionCalculatorV3). The formulas
are unchanged; what is fixed here is:

* the atmospheric-refraction branch, which used boolean-array indexing and so
  only worked for array input -- it is now :func:`numpy.select` and works for
  scalars and arrays alike;
* ``arccos`` arguments are clipped, which previously could produce NaN at the
  horizon;
* time is accepted as hours, not the error-prone "fraction of day" that led to
  ``Time = 10 / 24  # Noon`` in both call sites.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_EXCEL_EPOCH_OFFSET = 366 - 693960 + 2415018.5


def _excel_julian_day(year: int, month: int, day: int) -> float:
    return _dt.date(year, month, day).toordinal() + _EXCEL_EPOCH_OFFSET


def _refraction_deg(elevation_deg):
    """Atmospheric refraction correction, degrees. Scalar- and array-safe."""
    el = np.asarray(elevation_deg, dtype=float)
    # tan() blows up at 0; the >=5 deg branch is the only one that uses it.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.tan(np.deg2rad(el))
        high = (58.1 / t - 0.07 / t**3 + 0.000086 / t**5) / 3600.0
        low = -20.772 / t / 3600.0
    near = (1735.0 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711)))) / 3600.0

    out = np.select(
        [el >= 85.0, el >= 5.0, el >= -0.575],
        [np.zeros_like(el), high, near],
        default=low,
    )
    return out if out.ndim else float(out)


def _solar_core(lat, lon, tz, year, month, day, hours):
    """Shared geometry. ``hours`` is local clock time in hours."""
    time_frac = np.asarray(hours, dtype=float) / 24.0

    jul_day = _excel_julian_day(year, month, day) + time_frac - tz / 24.0
    jc = (jul_day - 2451545.0) / 36525.0

    gmls = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    seoc = (
        np.sin(np.deg2rad(gmas)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(np.deg2rad(2 * gmas)) * (0.019993 - 0.000101 * jc)
        + np.sin(np.deg2rad(3 * gmas)) * 0.000289
    )

    stl = gmls + seoc
    sal = stl - 0.00569 - 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 * jc))

    moe = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    oc = moe + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * jc))

    declination = np.rad2deg(
        np.arcsin(np.clip(np.sin(np.deg2rad(oc)) * np.sin(np.deg2rad(sal)), -1.0, 1.0))
    )

    vy = np.tan(np.deg2rad(oc / 2.0)) ** 2
    eot = 4 * np.rad2deg(
        vy * np.sin(2 * np.deg2rad(gmls))
        - 2 * eeo * np.sin(np.deg2rad(gmas))
        + 4 * eeo * vy * np.sin(np.deg2rad(gmas)) * np.cos(2 * np.deg2rad(gmls))
        - 0.5 * vy**2 * np.sin(4 * np.deg2rad(gmls))
        - 1.25 * eeo**2 * np.sin(2 * np.deg2rad(gmas))
    )
    return time_frac, declination, eot


def sun_position(lat, lon, tz, year, month, day, hours):
    """Solar azimuth and elevation in degrees.

    Azimuth is measured clockwise from north; elevation includes the
    atmospheric refraction correction.
    """
    time_frac, decl, eot = _solar_core(lat, lon, tz, year, month, day, hours)

    tst = np.mod(time_frac * 1440.0 + eot + 4.0 * lon - 60.0 * tz, 1440.0)
    hour_angle = np.where(tst / 4.0 < 0.0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)

    cos_zen = np.clip(
        np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
        + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(hour_angle)),
        -1.0,
        1.0,
    )
    zenith = np.rad2deg(np.arccos(cos_zen))
    elevation_raw = 90.0 - zenith

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = (
            (np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(zenith))) - np.sin(np.deg2rad(decl))
        ) / (np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(zenith)))
    theta = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    azimuth = np.mod(
        180.0 + (1.0 - np.sign(hour_angle)) * 180.0 + np.sign(hour_angle) * theta, 360.0
    )
    elevation = elevation_raw + _refraction_deg(elevation_raw)

    if np.ndim(hours) == 0:
        return float(azimuth), float(elevation)
    return np.asarray(azimuth), np.asarray(elevation)


def sunrise_sunset(lat, lon, tz, year, month, day) -> tuple[float, float]:
    """Sunrise and sunset as local clock hours."""
    _, decl, eot = _solar_core(lat, lon, tz, year, month, day, 12.0)

    cos_ha = np.clip(
        np.cos(np.deg2rad(90.833)) / (np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)))
        - np.tan(np.deg2rad(lat)) * np.tan(np.deg2rad(decl)),
        -1.0,
        1.0,
    )
    ha_sunrise = np.rad2deg(np.arccos(cos_ha))

    solar_noon = (720.0 - 4.0 * lon - eot + tz * 60.0) / 1440.0
    sunrise = (solar_noon * 1440.0 - ha_sunrise * 4.0) / 1440.0
    sunset = (solar_noon * 1440.0 + ha_sunrise * 4.0) / 1440.0
    return float(sunrise) * 24.0, float(sunset) * 24.0


def declination_hour_angle(lat, lon, tz, year, month, day, hours):
    """Solar declination and hour angle, degrees.

    These are the natural coordinates for interpolating optical efficiency: for
    a fixed site the sun direction is a bijection with (declination, hour
    angle), declination depends only on date, and hour angle only on clock time.
    Interpolating on this pair therefore extends a few traced days to the whole
    year without the day-length distortion that interpolating on clock hour
    would introduce.
    """
    time_frac, decl, eot = _solar_core(lat, lon, tz, year, month, day, hours)
    tst = np.mod(time_frac * 1440.0 + eot + 4.0 * lon - 60.0 * tz, 1440.0)
    hour_angle = np.where(tst / 4.0 < 0.0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    if np.ndim(hours) == 0:
        return float(decl), float(hour_angle)
    return np.broadcast_to(np.asarray(decl), np.shape(hour_angle)), np.asarray(hour_angle)


def hours_of_year(cfg, year: int) -> "pd.DataFrame":
    """Every hour of ``year`` with its sun position, declination and hour angle.

    This is the grid the annual energy integral runs on.
    """
    import pandas as pd

    site = cfg.site
    hours = np.arange(24.0)
    frames = []
    for doy in range(_days(year)):
        date = _dt.date(year, 1, 1) + _dt.timedelta(days=doy)
        az, el = sun_position(
            site.latitude,
            site.longitude,
            site.timezone,
            date.year,
            date.month,
            date.day,
            hours,
        )
        dec, ha = declination_hour_angle(
            site.latitude,
            site.longitude,
            site.timezone,
            date.year,
            date.month,
            date.day,
            hours,
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "hour": hours,
                    "solar_az_deg": az,
                    "solar_el_deg": el,
                    "declination_deg": dec,
                    "hour_angle_deg": ha,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _days(year: int) -> int:
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def _hhmm(hour: float) -> tuple[int, int]:
    """Round an hour-of-day to whole minutes, rolling 59.5min -> next hour."""
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return h, m


@dataclass(frozen=True)
class TimeStep:
    """One instant in the sweep."""

    date: _dt.date
    hour: float
    solar_az_deg: float
    solar_el_deg: float

    @property
    def solar_ze_deg(self) -> float:
        return 90.0 - self.solar_el_deg

    @property
    def key(self) -> str:
        """Filename-safe identifier, e.g. ``20260320_0730``."""
        h, m = _hhmm(self.hour)
        return f"{self.date:%Y%m%d}_{h:02d}{m:02d}"

    @property
    def label(self) -> str:
        h, m = _hhmm(self.hour)
        return f"{self.date:%Y-%m-%d} {h:02d}:{m:02d}"

    def __str__(self) -> str:
        return f"{self.label} (az {self.solar_az_deg:.1f}, el {self.solar_el_deg:.1f})"


def _solar_noon_hours(lat, lon, tz, year, month, day) -> float:
    """Local clock hour of true solar noon (hour angle = 0).

    Shares :func:`sunrise_sunset`'s own approximation: the equation of time
    is evaluated at clock hour 12 rather than at the actual noon instant,
    which it barely moves within a day.
    """
    _, _, eot = _solar_core(lat, lon, tz, year, month, day, 12.0)
    solar_noon = (720.0 - 4.0 * lon - eot + tz * 60.0) / 1440.0
    return float(solar_noon) * 24.0


def _elevation_edge_hour(
    lat, lon, tz, year, month, day, threshold_deg: float, horizon_hour: float, noon_hour: float
) -> float:
    """Local hour between ``horizon_hour`` (sunrise or sunset) and
    ``noon_hour`` (true solar noon) where elevation crosses ``threshold_deg``.

    Elevation is a strictly decreasing function of |hour angle|, and hour
    angle increases monotonically with clock time, so elevation is monotonic
    across the whole of either branch -- this bisects that branch, and
    ``horizon_hour``/``noon_hour`` can come in either clock order (the
    sunset branch runs noon -> set, a *decreasing* elevation as hour
    increases; the sunrise branch runs rise -> noon, increasing).

    Two degenerate cases return an edge unchanged rather than searching:

    * the sun never reaches ``threshold_deg`` this day (elevation at noon is
      still below it) -- returns ``noon_hour``, which collapses the caller's
      window to zero width (no timestep on this date can meet the floor);
    * the threshold is already met at ``horizon_hour`` (a threshold at or
      below the ~-0.833 deg sunrise/sunset elevation) -- returns
      ``horizon_hour`` unchanged, i.e. non-binding, so a ``min_elevation_deg``
      that is not actually stricter than the horizon has no effect.
    """
    _, noon_el = sun_position(lat, lon, tz, year, month, day, noon_hour)
    if noon_el < threshold_deg:
        return noon_hour
    _, horizon_el = sun_position(lat, lon, tz, year, month, day, horizon_hour)
    if horizon_el >= threshold_deg:
        return horizon_hour

    below, above = horizon_hour, noon_hour
    for _ in range(40):
        mid = 0.5 * (below + above)
        _, el_mid = sun_position(lat, lon, tz, year, month, day, mid)
        if el_mid < threshold_deg:
            below = mid
        else:
            above = mid
    return 0.5 * (below + above)


def elevation_floor_edges(cfg, date: _dt.date) -> tuple[float, float]:
    """The day's sunrise/sunset hours, narrowed by ``cfg.sweep.min_elevation_deg``.

    Deliberately separate from :func:`build_time_grid`'s own window (which
    also subtracts ``sunrise_margin_min`` on top of this): this is "where
    the elevation floor's own definition of negligible power actually
    starts holding", not the sampling window itself, and both
    :func:`build_time_grid` and :func:`heliostat.energy.traced_day_energy`
    call it so they agree on that point. ``traced_day_energy`` anchors its
    integral to zero power at whatever ``rise``/``set_`` this returns: if it
    anchored at the true horizon instead (:func:`sunrise_sunset`) while the
    floor has trimmed the actual samples back much further, the straight
    line from true-zero to the first surviving (already-well-above-floor)
    sample would cut across a real, non-linear power ramp and can OVERSTATE
    the day's energy -- this was measured to happen (a 5 deg floor scoring
    a higher ``traced_day_energy`` total than a 0 deg one on a real trace),
    exactly the "biases the integral" failure mode this module's docstrings
    warn about elsewhere. Anchoring at the elevation-floor edge instead
    makes that anchor point coincide with (up to the small
    ``sunrise_margin_min`` sliver) the first/last real sample, so there is
    no untraced gap left to mis-shape -- the excluded band is simply
    dropped, which only ever *removes* energy, never adds it.

    No ``min_elevation_deg`` on ``cfg.sweep`` -- or no ``cfg.sweep`` at all,
    e.g. a cfg built only for its ``site`` (see
    ``heliostat.energy.traced_day_energy``'s callers that pass one) --
    returns the plain :func:`sunrise_sunset` unchanged.
    """
    site = cfg.site
    rise, set_ = sunrise_sunset(
        site.latitude, site.longitude, site.timezone, date.year, date.month, date.day
    )
    sweep = getattr(cfg, "sweep", None)
    min_elevation_deg = getattr(sweep, "min_elevation_deg", None)
    if min_elevation_deg is None:
        return rise, set_
    noon = _solar_noon_hours(
        site.latitude, site.longitude, site.timezone, date.year, date.month, date.day
    )
    rise_edge = _elevation_edge_hour(
        site.latitude, site.longitude, site.timezone, date.year, date.month, date.day,
        min_elevation_deg, rise, noon,
    )
    set_edge = _elevation_edge_hour(
        site.latitude, site.longitude, site.timezone, date.year, date.month, date.day,
        min_elevation_deg, set_, noon,
    )
    return rise_edge, set_edge


def _sample_hours(rise: float, set_: float, margin: float, step: float) -> np.ndarray:
    """Uniform samples across ``[rise + margin, set_ - margin]``.

    ``step`` is the *maximum* spacing: the window is divided into
    ``ceil(span / step)`` equal intervals, so the true spacing is <= step and
    both window edges are always included. Returns an empty array if the
    margin has eaten the whole day.
    """
    start = rise + margin
    end = set_ - margin
    span = end - start
    if span <= 0:
        return np.empty(0, dtype=float)
    n_points = max(2, int(np.ceil(span / step)) + 1)
    return np.linspace(start, end, n_points)


def build_time_grid(cfg, dates: Iterable[_dt.date] | None = None) -> list[TimeStep]:
    """Timesteps sampled uniformly from ``sunrise + margin`` to ``sunset - margin``.

    ``hour_step`` is the *maximum* allowed spacing, not a clock grid: the
    daylight window is divided into equal intervals no wider than
    ``hour_step``, so the first and last samples always land exactly on the
    window edges instead of being snapped inward to the nearest whole hour.

    Earlier versions snapped to the ``hour_step`` clock grid (e.g. samples
    only at 07:00, 08:00, ...) so the same clock times would recur across
    dates and through-the-day figures stayed comparable frame to frame. That
    snapping could discard most of an hour at each end of the day -- a 0.03 h
    overshoot past a whole hour lost a full hour of margin -- which biased
    the collected-power integral this grid feeds low. Since the sweep is used
    to integrate power into MW-hr, capturing the true sunrise/sunset edges of
    the day matters more than clock alignment across dates. Do not restore
    the snapping without first fixing the integral to correct for the
    resulting sample-weight distortion.

    ``cfg.sweep.min_elevation_deg`` -- an optional elevation floor
    -----------------------------------------------------------------------
    Low-sun timesteps (a few degrees above the horizon) cost the same trace
    time as a noon one but collect almost no power, both because DNI itself
    falls off at high air mass and because cosine incidence collapses as the
    sun approaches the mirror plane. ``cfg.sweep.min_elevation_deg``, if the
    ``sweep`` namespace carries it, excludes them -- but it does NOT simply
    drop samples out of the margin-trimmed window above and leave the
    remaining ones' spacing/weight alone, which would silently redistribute
    the excluded steps' (small but nonzero) share of the integral onto their
    neighbours and bias the MW-hr total high. Instead :func:`elevation_floor_edges`
    shrinks the window itself for each date -- the clock hour where
    elevation crosses ``min_elevation_deg`` on the morning and evening
    branches replaces ``rise``/``set_`` as the bounds fed to
    ``_sample_hours`` -- i.e. exactly the same "shrink the integration
    window, keep the equal-interval grid logic" move ``sunrise_margin_min``
    already makes, just anchored to an elevation crossing instead of a fixed
    number of minutes. A trapezoid integrator over just the surviving
    samples (e.g. ``heliostat.web.app._day_energy_kwh``) then simply
    integrates a narrower, still-equal-interval domain -- since collected
    power is never negative, that can only remove energy, never add it.
    :func:`heliostat.energy.traced_day_energy` anchors its own trapezoid to
    zero power at :func:`elevation_floor_edges`' bounds rather than the true
    horizon for the same reason -- see that function's docstring.

    Missing the attribute entirely (any ``cfg`` built before this floor
    existed, e.g. the paper-reproduction example's ``paper_cfg``) is treated
    as "no floor" and reproduces prior behaviour exactly -- this is a
    default via ``getattr``, not a dataclass field, precisely so old cfg
    objects need no changes.
    """
    site = cfg.site
    step = cfg.sweep.hour_step
    margin = cfg.sweep.sunrise_margin_min / 60.0
    dates = tuple(dates) if dates is not None else cfg.sweep.dates

    steps: list[TimeStep] = []
    for date in dates:
        rise_edge, set_edge = elevation_floor_edges(cfg, date)
        for hour in _sample_hours(rise_edge, set_edge, margin, step):
            hour = float(hour)
            az, el = sun_position(
                site.latitude,
                site.longitude,
                site.timezone,
                date.year,
                date.month,
                date.day,
                hour,
            )
            steps.append(TimeStep(date=date, hour=hour, solar_az_deg=az, solar_el_deg=el))
    return steps


def describe_time_grid(cfg, dates: Iterable[_dt.date] | None = None) -> str:
    """Human-readable summary of the sweep grid, for logs and sanity checks."""
    site = cfg.site
    dates = tuple(dates) if dates is not None else cfg.sweep.dates
    steps = build_time_grid(cfg, dates)
    lines = [
        f"Site: lat {site.latitude}, lon {site.longitude}, TZ {site.timezone:+d}",
        f"{len(steps)} timesteps across {len(dates)} dates",
    ]
    for date in dates:
        rise, set_ = sunrise_sunset(
            site.latitude, site.longitude, site.timezone, date.year, date.month, date.day
        )
        day = [s for s in steps if s.date == date]
        hours = ", ".join(f"{s.hour:g}" for s in day)
        peak = max(day, key=lambda s: s.solar_el_deg) if day else None
        if peak:
            spacing_min = (
                (day[-1].hour - day[0].hour) / (len(day) - 1) * 60.0 if len(day) > 1 else 0.0
            )
            start_h, start_m = _hhmm(day[0].hour)
            end_h, end_m = _hhmm(day[-1].hour)
            lines.append(
                f"  {date:%Y-%m-%d}  sunrise {rise:5.2f}  sunset {set_:5.2f}  "
                f"window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}  "
                f"{len(day):2d} steps @ {spacing_min:5.2f} min  "
                f"peak el {peak.solar_el_deg:5.2f} deg at {peak.hour:g}h"
            )
        else:
            lines.append(f"  {date:%Y-%m-%d}  no daylight steps")
        lines.append(f"      hours: {hours}")
    return "\n".join(lines)
