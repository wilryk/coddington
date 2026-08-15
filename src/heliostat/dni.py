"""Direct normal irradiance by date and hour.

DNI never affects the ray trace: a trace source is normalised so its aperture
delivers exactly 1000 W/m^2, so real DNI enters as a scale factor at analysis
time:

    flux = count * W_per_ray * throughput * eta_occlusion
           * (dni / 1000) / bin_area

``eta_occlusion`` is the union of shading and blocking. That means the DNI
model can be swapped, refined, or replaced with measured data at any point
**without re-tracing anything**.

Providers
---------
``ConstantDNI``        fixed value; what reproduces prior work.
``TableDNI``           measured or downloaded hourly series, matched by exact
                        calendar day. Carries real day-to-day weather, which
                        means it also carries whatever weather the TMY splice
                        happened to donate for that one day (see
                        ``MonthlyProfileDNI``).
``MonthlyProfileDNI``   mean diurnal curve per calendar month, averaged over
                        every day in that month. A good default when a site's
                        DNI table is a single typical-meteorological-year
                        splice rather than a multi-year record: it resolves
                        the season without over-fitting whatever weather that
                        one synthetic year happened to carry on any given day.

Two free online sources are supported by :func:`fetch`, neither needing an API
key:

``pvgis``  EU JRC PVGIS typical meteorological year. Purpose-built for annual
           energy estimates, global coverage, returns Gb(n) = DNI.
``nasa``   NASA POWER hourly ALLSKY_SFC_SW_DNI for a specific historical year.

Both return UTC timestamps, which are converted to the site's local clock time
using the configured timezone offset.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_DNI = 1000.0  # W/m^2, the DNI a trace is normalised to


class DNIProvider(ABC):
    """Returns DNI in W/m^2 for a given date and local clock hour."""

    @abstractmethod
    def dni(self, date: _dt.date, hour: float) -> float: ...

    def scale(self, date: _dt.date, hour: float) -> float:
        """Multiplier to apply to trace-derived flux."""
        return self.dni(date, hour) / STANDARD_DNI

    def describe(self) -> str:
        return self.__class__.__name__


class ConstantDNI(DNIProvider):
    def __init__(self, value: float = STANDARD_DNI):
        self.value = float(value)

    def dni(self, date: _dt.date, hour: float) -> float:
        return self.value

    def describe(self) -> str:
        return f"ConstantDNI({self.value:g} W/m2)"


class TableDNI(DNIProvider):
    """Hourly DNI from a table, matched by day-of-year and hour.

    The table is matched on (month, day, hour) rather than absolute date, so a
    TMY or any single historical year can drive a sweep configured for any year.
    Hours are linearly interpolated; missing days fall back to ``default``.
    """

    def __init__(self, frame: pd.DataFrame, default: float = STANDARD_DNI, source: str = ""):
        required = {"month", "day", "hour", "dni_w_m2"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"DNI table missing columns: {sorted(missing)}")
        self.default = float(default)
        self.source = source
        self._by_day: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for (m, d), grp in frame.groupby(["month", "day"], sort=False):
            grp = grp.sort_values("hour")
            self._by_day[(int(m), int(d))] = (
                grp["hour"].to_numpy(float),
                grp["dni_w_m2"].to_numpy(float),
            )

    def dni(self, date: _dt.date, hour: float) -> float:
        entry = self._by_day.get((date.month, date.day))
        if entry is None:
            return self.default
        hours, values = entry
        if hours.size == 1:
            return float(values[0])
        return float(np.interp(hour, hours, values, left=values[0], right=values[-1]))

    def describe(self) -> str:
        return f"TableDNI({len(self._by_day)} days from {self.source or 'table'})"


class MonthlyProfileDNI(DNIProvider):
    """Mean diurnal DNI curve per calendar month, averaged across that month's days.

    Built by grouping the source table on ``(month, hour)`` and averaging over
    every row sharing that key -- deliberately *not* built by first averaging
    each day down to a scalar and then averaging days together, and not keyed
    on individual calendar days at query time at all. Hours are interpolated
    within the resulting profile exactly as :class:`TableDNI` interpolates
    within a day.

    The ``(month, hour)`` grouping also sidesteps a TMY month-splice stub for
    free: (month=2, day=29) can be a leftover few-row bucket produced when a
    Feb/Mar splice boundary walks past the end of February in a leap donor
    year (see :class:`TableDNI`), and the matching (2, 28) bucket is short by
    those same rows. Grouping by day first and then averaging days would have
    to special-case this (a near-empty "day" skews a per-day mean, and naively
    averaging 29 day-means for a 28-day month is wrong). Grouping by (month,
    hour) instead needs no such logic: at the affected hours the (2, 29) stub
    simply supplies the count that (2, 28) is missing, so every hour of
    February is averaged over exactly 28 rows -- the true day count --
    regardless of which day-bucket those rows happen to live in.
    """

    def __init__(self, frame: pd.DataFrame, default: float = STANDARD_DNI, source: str = ""):
        required = {"month", "day", "hour", "dni_w_m2"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"DNI table missing columns: {sorted(missing)}")
        self.default = float(default)
        self.source = source
        self._by_month: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for m, grp in frame.groupby("month", sort=False):
            profile = grp.groupby("hour")["dni_w_m2"].mean().sort_index()
            self._by_month[int(m)] = (
                profile.index.to_numpy(float),
                profile.to_numpy(float),
            )

    def dni(self, date: _dt.date, hour: float) -> float:
        entry = self._by_month.get(date.month)
        if entry is None:
            return self.default
        hours, values = entry
        if hours.size == 1:
            return float(values[0])
        return float(np.interp(hour, hours, values, left=values[0], right=values[-1]))

    def day_kwh_m2(self, month: int) -> float:
        """Day-integrated energy under this month's average diurnal curve."""
        hours, values = self._by_month[month]
        return float(np.trapz(values, hours) / 1000.0)

    def describe(self) -> str:
        return f"MonthlyProfileDNI({len(self._by_month)} months from {self.source or 'table'})"


class DailyClimatologyDNI(DNIProvider):
    """Mean diurnal curve per calendar DAY, averaged over many real years.

    The step up from :class:`MonthlyProfileDNI`. That one averages a single
    synthetic year (a TMY splices twelve donor years) within each month, so it
    resolves the season only twelve ways and inherits whichever weather each
    donor month happened to carry. This averages many *real* years at daily
    resolution, which is what a per-day figure needs to be defensible.

    Two averaging stages, and the second is not cosmetic
    ---------------------------------------------------
    1. group by (day-of-year, hour) across years -- with e.g. a 24-year record
       that is 24 samples per point, and 24 samples of a variable sky is still
       visibly noisy day to day;
    2. a circular +/-``window`` day mean over day-of-year, which raises that to
       ~N x (2*window+1) samples -- about 260 at the default +/-5 days with a
       24-year record -- without touching the diurnal shape, since the
       smoothing runs along the day axis only.

    The window is deliberately narrow: at most sites the seasonal DNI cycle
    turns over across months, so a few days cannot flatten anything real,
    while it removes most of the sampling noise. It is a parameter rather than
    a constant so the choice can be shown to be uncritical.

    February 29 folds onto February 28 rather than being dropped: leap days
    are real irradiance and there is no 366th slot to report them in.
    """

    #: Day-of-year for the first of each month in a NON-leap year.
    _MONTH_START = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])

    def __init__(
        self,
        frame: pd.DataFrame,
        default: float = STANDARD_DNI,
        source: str = "",
        window_days: int = 5,
    ):
        required = {"month", "day", "hour", "dni_w_m2"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"DNI table missing columns: {sorted(missing)}")
        self.default = float(default)
        self.source = source
        self.window_days = int(window_days)
        self.n_years = int(frame["year"].nunique()) if "year" in frame else 1

        doy = self._doy(frame["month"].to_numpy(int), frame["day"].to_numpy(int))
        work = pd.DataFrame(
            {
                "doy": doy,
                "hour": frame["hour"].to_numpy(float),
                "dni": frame["dni_w_m2"].to_numpy(float),
            }
        )

        # (doy, hour) grid: rows are days, columns the common hour stamps.
        grid = work.groupby(["doy", "hour"])["dni"].mean().unstack("hour").sort_index()
        grid = grid.reindex(range(1, 366))
        # Hours are on a fixed stamp set; a missing cell means no sample, and
        # night is a real zero, so interpolate along the day axis only.
        grid = grid.interpolate(axis=0, limit_direction="both")

        self._hours = grid.columns.to_numpy(float)
        values = grid.to_numpy(float)
        w = self.window_days
        if w > 0:
            # Circular, so 1 January averages with late December.
            padded = np.concatenate([values[-w:], values, values[:w]], axis=0)
            kernel = np.ones(2 * w + 1) / (2 * w + 1)
            values = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="valid"), 0, padded
            )
        self._values = values
        self._n_per_point = self.n_years * (2 * self.window_days + 1)

    @classmethod
    def _doy(cls, month, day):
        """Day-of-year on a non-leap calendar; 29 Feb folds onto 28 Feb."""
        month = np.asarray(month, int)
        day = np.minimum(np.asarray(day, int), np.where(month == 2, 28, 31))
        return cls._MONTH_START[month - 1] + day

    def dni(self, date: _dt.date, hour: float) -> float:
        doy = int(self._doy(date.month, date.day))
        row = self._values[doy - 1]
        if not np.isfinite(row).any():
            return self.default
        return float(np.interp(hour, self._hours, row, left=row[0], right=row[-1]))

    def day_kwh_m2(self, date: _dt.date) -> float:
        """Day-integrated energy under this day's average diurnal curve."""
        row = self._values[int(self._doy(date.month, date.day)) - 1]
        return float(np.trapz(row, self._hours) / 1000.0)

    def annual_kwh_m2(self) -> float:
        """Sum of the 365 daily integrals -- the climatology's annual DNI."""
        return float(sum(np.trapz(self._values[i], self._hours) / 1000.0 for i in range(365)))

    def describe(self) -> str:
        return (
            f"DailyClimatologyDNI({self.n_years} years, +/-"
            f"{self.window_days} d window, ~{self._n_per_point} samples per "
            f"day-hour, from {self.source or 'table'})"
        )


def _tidy(frame: pd.DataFrame, tz_offset_hours: float) -> pd.DataFrame:
    """Convert a UTC timestamp + DNI frame into month/day/hour local form."""
    local = frame["timestamp_utc"] + pd.to_timedelta(tz_offset_hours, unit="h")
    return pd.DataFrame(
        {
            "month": local.dt.month,
            "day": local.dt.day,
            "hour": local.dt.hour + local.dt.minute / 60.0,
            "dni_w_m2": frame["dni_w_m2"].to_numpy(float),
        }
    )


class ClearSkyDNI(DNIProvider):
    """Cloud-free DNI from solar elevation alone. No weather, no measurement.

    Why this exists
    ---------------
    Every other provider carries a site's weather, and at low latitudes that
    weather can be the dominant signal -- a rainy-season month can lose a
    third of its DNI to convective cloud between two clock hours, and the
    annual total can swing widely across years. That is real and belongs in
    an energy figure, but it buries the optics -- two layouts can only be
    compared on collection efficiency if the light arriving is identical and
    smooth.

    This provider gives that: a clear, dry, cloud-free sky where DNI depends on
    nothing but how much atmosphere the beam crosses.

    The model
    ---------
    Beer-Lambert through a plane-parallel atmosphere, in the Meinel form::

        DNI = E0 * tau ** (AM ** 0.678)

    ``AM`` is the relative optical air mass -- 1.0 with the sun overhead, ~2 at
    30 deg elevation, ~38 at the horizon -- computed with Kasten & Young (1989)
    rather than 1/sin(h), which diverges at sunrise. The 0.678 exponent is
    Meinel's empirical correction for the fact that attenuation is not a pure
    exponential in air mass.

    ``tau`` is not set directly. It is derived from ``am1_w_m2``, the DNI you
    want at exactly one air mass, because that is the number a person actually
    has an opinion about: 1361 W/m2 arrives above the atmosphere and one air
    mass of clean dry air takes it to about 1000 at sea level, which is where
    the "1000 W/m2" convention comes from and why traces are normalised to it.

    What it is NOT
    --------------
    Not a measurement and not a TMY. Annual energy computed against it is an
    upper bound -- the yield of a field at this site if it never clouded
    over -- and should be labelled that way rather than compared against the
    climatology numbers as though they were alternatives.
    """

    #: Solar constant, W/m2 (mean earth-sun distance).
    E0 = 1361.0

    def __init__(self, site, am1_w_m2: float = 1000.0, e0_w_m2: float = E0):
        self.site = site
        self.am1 = float(am1_w_m2)
        self.e0 = float(e0_w_m2)
        if not 0.0 < self.am1 < self.e0:
            raise ValueError(
                f"clearsky am1_w_m2 must be between 0 and the solar constant "
                f"{self.e0:g} W/m2; got {self.am1:g}"
            )
        self.tau = self.am1 / self.e0

    @staticmethod
    def air_mass(elevation_deg: float) -> float:
        """Kasten & Young (1989) relative optical air mass.

        1/sin(h) is the textbook form and is wrong where it matters most: it
        goes to infinity at h = 0, so a sunrise timestep would get zero DNI by
        singularity rather than by physics. This stays finite (~38 at the
        horizon).
        """
        if elevation_deg <= 0.0:
            return float("inf")
        h = float(elevation_deg)
        return 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)

    def dni(self, date: _dt.date, hour: float) -> float:
        from . import solar

        _az, el = solar.sun_position(
            self.site.latitude,
            self.site.longitude,
            self.site.timezone,
            date.year,
            date.month,
            date.day,
            hour,
        )[:2]
        if el <= 0.0:
            return 0.0
        am = self.air_mass(el)
        if not np.isfinite(am):
            return 0.0
        return float(self.e0 * self.tau ** (am**0.678))

    def day_kwh_m2(self, date: _dt.date, step_h: float = 0.25) -> float:
        hours = np.arange(0.0, 24.0, step_h)
        return float(sum(self.dni(date, h) for h in hours) * step_h / 1000.0)

    def annual_kwh_m2(self, year: int = 2026) -> float:
        start = _dt.date(year, 1, 1)
        days = 366 if _dt.date(year, 12, 31).timetuple().tm_yday == 366 else 365
        return float(sum(self.day_kwh_m2(start + _dt.timedelta(days=i)) for i in range(days)))

    def describe(self) -> str:
        return (
            f"ClearSkyDNI(Meinel, {self.am1:g} W/m2 at air mass 1, "
            f"E0 {self.e0:g}, tau {self.tau:.4f}) -- cloud-free upper bound"
        )


_TABLE_MODES = {"table": TableDNI, "monthly": MonthlyProfileDNI, "climatology": DailyClimatologyDNI}

#: Every mode name provider_for accepts. Callers that validate a mode string
#: from their own configuration layer should validate against this set rather
#: than keeping a separate list of their own, so adding a mode here cannot
#: leave that validation behind.
ALL_MODES = {"constant", "clearsky"} | set(_TABLE_MODES)

#: Modes whose table is a MULTI-YEAR record rather than one year. They read a
#: different file from the single-year table, because pointing a climatology
#: at a single TMY year would build a 1-year "climatology" and say nothing
#: about it -- the describe() string would read "1 years" and be easy to miss.
_MULTIYEAR_MODES = {"climatology"}

#: Default relative path for the multi-year DNI record used by climatology
#: mode. Purely a fallback name -- nothing checks this path exists at import
#: time, and callers are expected to override it (either by passing
#: ``multiyear_file`` on their config object, or by supplying the table
#: DataFrame directly to :class:`DailyClimatologyDNI`). No such file ships
#: with this package.
_MULTIYEAR_FILE = "data/dni_nasa_hourly.csv"


class SolarTimeAligned(DNIProvider):
    """A DNI record from one longitude, applied at another.

    The irradiance table is indexed by CLOCK hour at the place it was measured;
    the optical efficiency is indexed by hour angle at the place that was
    traced. When those are different longitudes, pairing them by clock hour
    puts the two diurnal curves out of step -- both peak at their own solar
    noon, and solar noon happens at a different clock time at each.

    The correction is a shift of the lookup, not a fudge: the efficiency
    surface is keyed on declination and HOUR ANGLE, which do not depend on
    longitude at all, so evaluating the table at the traced site's solar time
    is exactly what re-tracing at the table's longitude would have produced.
    Latitude is the part that a shift cannot fix -- it changes the sun's path
    rather than its clock -- so this stays honest only while the two sites'
    latitudes are close enough to ignore.

    Positive ``lon_data - lon_site`` (data to the EAST) means the data's solar
    noon comes first, so the table must be read earlier in the day.
    """

    def __init__(self, inner: DNIProvider, lon_data_deg: float, lon_site_deg: float):
        self.inner = inner
        self.lon_data_deg = float(lon_data_deg)
        self.lon_site_deg = float(lon_site_deg)
        #: Hours to add to the site clock hour before reading the table.
        self.offset_h = -(self.lon_data_deg - self.lon_site_deg) / 15.0

    def dni(self, date: _dt.date, hour: float) -> float:
        return self.inner.dni(date, hour + self.offset_h)

    def describe(self) -> str:
        return (
            f"{self.inner.describe()} aligned {self.offset_h * 60:+.0f} min "
            f"(data {self.lon_data_deg:g} deg -> site {self.lon_site_deg:g} deg)"
        )


def provider_for(cfg, mode: str | None = None) -> DNIProvider:
    """Build a DNI provider by name from a config-like object.

    ``cfg`` is duck-typed: it needs a ``.site`` (with ``.longitude``) and,
    for any mode besides ``"constant"``/``"clearsky"``, a ``.dni`` section
    exposing ``mode``, ``constant_w_m2``, ``table_file``, and optionally
    ``clearsky_am1_w_m2``, ``multiyear_file``, ``data_longitude``. It also
    needs a ``.path(relative)`` method resolving a relative path to an
    absolute one -- this package does not prescribe how that lookup works.

    ``mode`` overrides ``cfg.dni.mode`` for this call only; pass ``None`` (the
    default) to build whatever the config's ``dni.mode`` currently says. This
    is the one-line switch a live control needs -- it takes the already-loaded
    ``cfg`` and a mode string, and never touches or reloads configuration, so
    flipping modes at runtime doesn't require a reload.
    """
    spec = getattr(cfg, "dni", None)
    mode = mode or (spec.mode if spec is not None else "constant")

    if mode == "constant":
        value = STANDARD_DNI if spec is None else spec.constant_w_m2
        return ConstantDNI(value)

    if mode == "clearsky":
        am1 = getattr(spec, "clearsky_am1_w_m2", 1000.0) if spec else 1000.0
        return ClearSkyDNI(cfg.site, am1_w_m2=am1)

    if mode in _TABLE_MODES:
        if spec is None:
            raise ValueError(f"DNI mode {mode!r} requires a dni section on cfg")
        table = (
            getattr(spec, "multiyear_file", None) or _MULTIYEAR_FILE
            if mode in _MULTIYEAR_MODES
            else spec.table_file
        )
        path = cfg.path(table)
        if not path.exists():
            hint = (
                "supply a multi-year hourly DNI table"
                if mode in _MULTIYEAR_MODES
                else "supply an hourly DNI table (see heliostat.dni.fetch)"
            )
            raise FileNotFoundError(
                f'DNI table {path} not found. {hint}, or set dni mode to "constant".'
            )
        if path.suffix.lower() in (".xlsx", ".xls"):
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path)
        provider = _TABLE_MODES[mode](frame, default=spec.constant_w_m2, source=path.name)

        # A table measured at a different longitude than the traces must be
        # read at the traced site's solar time, not its own clock. Absent the
        # setting nothing happens, so stored runs read exactly as before.
        lon_data = getattr(spec, "data_longitude", None)
        if lon_data is not None and abs(float(lon_data) - cfg.site.longitude) > 1e-9:
            provider = SolarTimeAligned(provider, float(lon_data), cfg.site.longitude)
        return provider

    raise ValueError(
        f"unknown DNI mode {mode!r}; use one of {sorted({'constant'} | set(_TABLE_MODES))}"
    )


def load_dni_provider(cfg) -> DNIProvider:
    """Build the provider described by ``cfg``'s ``dni`` section.

    Thin wrapper over :func:`provider_for` kept for existing call sites.
    """
    return provider_for(cfg)


# --------------------------------------------------------------------------
# Online fetchers. Network access only happens when these are called directly.
# --------------------------------------------------------------------------


def fetch(source: str, cfg, out_path: Path | None = None, year: int | None = None) -> Path:
    """Download an hourly DNI series and cache it as a tidy CSV.

    Neither source requires an API key. Returns the path written. ``cfg``
    only needs a ``.site`` (with ``.latitude``/``.longitude``/``.timezone``)
    and, if ``out_path`` is omitted, a ``.path(relative)`` method.
    """
    site = cfg.site
    out_path = Path(out_path) if out_path else cfg.path(f"dni_{source}.csv")

    if source == "pvgis":
        frame = _fetch_pvgis_tmy(site.latitude, site.longitude)
    elif source == "nasa":
        frame = _fetch_nasa_power(site.latitude, site.longitude, year)
    else:
        raise ValueError(f"unknown DNI source {source!r}; use 'pvgis' or 'nasa'")

    tidy = _tidy(frame, site.timezone)
    tidy.to_csv(out_path, index=False)
    return out_path


def _fetch_pvgis_tmy(lat: float, lon: float) -> pd.DataFrame:
    """PVGIS typical meteorological year. Column ``Gb(n)`` is DNI in W/m^2."""
    import io
    import urllib.request

    url = f"https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat={lat}&lon={lon}&outputformat=csv"
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("time(UTC)"))
    end = start + 1
    while end < len(lines) and lines[end] and lines[end][0].isdigit():
        end += 1
    table = pd.read_csv(io.StringIO("\n".join(lines[start:end])))

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(table["time(UTC)"], format="%Y%m%d:%H%M"),
            "dni_w_m2": table["Gb(n)"].to_numpy(float),
        }
    )


def _fetch_nasa_power(lat: float, lon: float, year: int | None) -> pd.DataFrame:
    """NASA POWER hourly all-sky DNI for one historical year."""
    import json
    import urllib.request

    if year is None:
        year = _dt.date.today().year - 1
    # time-standard MUST be stated. The hourly endpoint defaults to LST (local
    # solar time), so omitting it returns timestamps that are already local --
    # and _tidy then subtracts the site offset a second time. That shifts the
    # whole diurnal curve 3 h early here, which leaves the annual DNI total
    # untouched (a shift cannot change a sum) while misaligning every hour of
    # sunlight against the field's optical efficiency. Ask for UTC so the
    # column name is true and _tidy's conversion is the only one.
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        "?parameters=ALLSKY_SFC_SW_DNI&community=RE&time-standard=UTC"
        f"&latitude={lat}&longitude={lon}"
        f"&start={year}0101&end={year}1231&format=JSON"
    )
    with urllib.request.urlopen(url, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    series = payload["properties"]["parameter"]["ALLSKY_SFC_SW_DNI"]
    stamps = pd.to_datetime(list(series.keys()), format="%Y%m%d%H")
    values = np.array(list(series.values()), dtype=float)
    values[values < -900] = 0.0  # POWER fill value
    return pd.DataFrame({"timestamp_utc": stamps, "dni_w_m2": values})
