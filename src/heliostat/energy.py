"""Annual energy collected by the field.

The key decomposition
---------------------
Optical efficiency depends **only on the sun's direction**, not on the calendar.
Two dates that put the sun at the same (declination, hour angle) give identical
optics. DNI, by contrast, varies hour to hour and is cheap to obtain.

So the expensive ray trace is used to build a dimensionless surface

    eta_optical(declination, hour_angle)
        = power delivered to the receiver / (DNI * total mirror area)

sampled at the traced timesteps, and the annual integral is then run over all
8760 hours of the year using a full DNI series:

    E_year = sum over hours [ eta_optical(dec_h, ha_h) * DNI_h * A_mirror * dt ]

This is why a handful of traced days yields a genuine annual number rather than
four days multiplied by 91.

Sampling caveat
---------------
Interpolation happens across *declination*. Four dates spanning both solstices
and an equinox give only a few distinct declinations, so the declination axis
is interpolated from a sparse set of points spanning -23.4..+23.4 deg. That is
enough for a defensible estimate but coarse. A dozen dates through the year
give a dozen declinations and a much better surface, at proportionally more
trace cost. :func:`declination_coverage` reports what a given date set
actually spans.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from . import solar

STANDARD_DNI = 1000.0


def optical_efficiency(summary: pd.DataFrame, cfg, n_heliostats: int | None = None) -> pd.DataFrame:
    """Per-timestep field optical efficiency, dimensionless.

    Aggregates the per-heliostat summary rows into one row per timestep. The
    result is DNI-independent by construction, which is what allows it to be
    reused across every hour of the year.
    """
    n_heliostats = n_heliostats or summary["heliostat_id"].nunique()
    mirror_area_total = cfg.field.mirror_area_m2 * n_heliostats

    grouped = summary.groupby(["date", "hour"], as_index=False).agg(
        power_w=("power_w", "sum"),
        solar_az_deg=("solar_az_deg", "first"),
        solar_el_deg=("solar_el_deg", "first"),
        n_heliostats=("heliostat_id", "nunique"),
    )
    # power_w in the summary is computed at STANDARD_DNI, so dividing by
    # (STANDARD_DNI * area) removes the irradiance and leaves pure optics.
    grouped["eta_optical"] = grouped["power_w"] / (STANDARD_DNI * mirror_area_total)

    dec, ha = [], []
    for _, row in grouped.iterrows():
        d = row["date"] if isinstance(row["date"], _dt.date) else pd.to_datetime(row["date"]).date()
        dd, hh = solar.declination_hour_angle(
            cfg.site.latitude,
            cfg.site.longitude,
            cfg.site.timezone,
            d.year,
            d.month,
            d.day,
            float(row["hour"]),
        )
        dec.append(dd)
        ha.append(hh)
    grouped["declination_deg"] = dec
    grouped["hour_angle_deg"] = ha
    return grouped


#: Obliquity of the ecliptic. The declination the sun turns around at, and
#: therefore the axis the fit is mirrored about.
DEC_TURNING_POINT = 23.4392

#: Declination gap, degrees, that separates one traced DAY from the next.
#: Within a day declination drifts a few tenths of a degree; between traced
#: dates the gap is typically several degrees or more. Anything in between
#: works; 1.0 is comfortably inside both margins.
_DAY_SPLIT_DEG = 1.0

#: Points across the normalised day. Dense enough that linear interpolation
#: along it is negligible against the ray-trace noise, cheap enough to build
#: per call.
_U_GRID = np.linspace(-1.0, 1.0, 241)


def _sunset_hour_angle(lat_deg: float, dec_deg):
    """|H0| where the sun's elevation is zero, degrees."""
    x = -np.tan(np.radians(lat_deg)) * np.tan(np.radians(np.asarray(dec_deg, float)))
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def build_interpolator(efficiency: pd.DataFrame, latitude_deg: float):
    """Interpolate eta_optical over (declination, hour angle). No convex hull.

    Why not a triangulation
    -----------------------
    An earlier form of this function triangulated the scattered traced points
    and fell back to nearest-neighbour outside the hull. Declination is not
    constant within a day -- it drifts across the traced hours -- so the
    points at the extreme declination cover only part of that day's hour-angle
    range, the hull's top edge comes out ragged, and both SOLSTICE afternoons
    fall outside it and switch method. Measured against a clear-sky DNI, where
    the daily curve is analytic and must be smooth, that put a several-MWh
    step in the year at both solstices. It was pure artefact.

    The two ideas here
    ------------------
    **Normalised hour angle.** Each traced day is resampled onto
    ``u = ha / H0(declination)``, zero at solar noon and +/-1 at true sunrise
    and sunset. Every day then covers exactly the same interval regardless of
    length, so a short winter day and a long summer one overlap completely
    instead of partially -- which is what removes the hull problem rather than
    patching it. Each day's curve is anchored to zero at |u| = 1, because with
    the sun on the horizon the field collects nothing.

    **Mirror at the turning points.** Declination genuinely reverses at
    +/-23.44 deg, so the sample set is reflected about both extremes before
    fitting. PCHIP at an endpoint has only a one-sided slope estimate; the
    mirror gives it the symmetric neighbourhood the physics actually has, at
    exactly the two days that were wrong.

    PCHIP across declination because that axis is sparse -- a handful of
    samples for a whole year -- and shape matters there; it is
    monotone-preserving, so unlike a natural cubic it cannot overshoot into an
    efficiency the trace never saw.

    Returns ``(values, extrapolated)``; the second is True only where a query
    declination lies outside the traced range, which cannot happen for a real
    year once both solstices are traced.
    """
    need = {"declination_deg", "hour_angle_deg", "eta_optical"}
    missing = need - set(efficiency.columns)
    if missing:
        raise ValueError(f"efficiency frame missing {sorted(missing)}")

    dec_all = efficiency["declination_deg"].to_numpy(float)
    ha_all = efficiency["hour_angle_deg"].to_numpy(float)
    eta_all = efficiency["eta_optical"].to_numpy(float)

    order = np.argsort(dec_all)
    dec_all, ha_all, eta_all = dec_all[order], ha_all[order], eta_all[order]

    # Split into traced DAYS on declination gaps. Grouping by a date column
    # would be cleaner but callers do not all have one -- some frames are
    # built directly from per-timestep power without carrying a calendar date.
    breaks = np.where(np.diff(dec_all) > _DAY_SPLIT_DEG)[0] + 1
    groups = np.split(np.arange(len(dec_all)), breaks)

    n_dec = len(groups)
    if len(dec_all) < 4 or n_dec < 2:
        raise ValueError(
            f"Annual interpolation needs at least 2 distinct declinations and 4 "
            f"timesteps; got {n_dec} declination(s) across {len(dec_all)} "
            f"timestep(s). Trace more dates -- see energy.suggest_sweep_dates()."
        )
    if n_dec < 3:
        import warnings

        warnings.warn(
            f"Only {n_dec} distinct declinations traced. The annual total will be "
            f"a crude extrapolation; 6-10 declinations is a reasonable target.",
            stacklevel=2,
        )

    day_dec, rows = [], []
    for g in groups:
        dec = float(dec_all[g].mean())
        h0 = float(_sunset_hour_angle(latitude_deg, dec))
        u = np.concatenate(([-1.0], ha_all[g] / h0, [1.0]))
        eta = np.concatenate(([0.0], eta_all[g], [0.0]))
        s = np.argsort(u)
        u, eta = u[s], eta[s]
        keep = np.concatenate(([True], np.diff(u) > 1e-9))
        rows.append(np.interp(_U_GRID, u[keep], eta[keep]))
        day_dec.append(dec)

    day_dec = np.asarray(day_dec, float)
    M = np.vstack(rows)

    d = DEC_TURNING_POINT
    dec_ext = np.concatenate((-2 * d - day_dec[::-1], day_dec, 2 * d - day_dec[::-1]))
    M_ext = np.vstack((M[::-1], M, M[::-1]))
    keep = np.concatenate(([True], np.diff(dec_ext) > 1e-9))
    across_dec = PchipInterpolator(dec_ext[keep], M_ext[keep], axis=0, extrapolate=True)

    lo, hi = day_dec.min(), day_dec.max()

    def interp(dec, ha):
        dec = np.asarray(dec, float)
        ha = np.asarray(ha, float)
        u = ha / _sunset_hour_angle(latitude_deg, dec)
        cols = across_dec(dec)
        idx = np.clip(np.searchsorted(_U_GRID, u) - 1, 0, len(_U_GRID) - 2)
        u0, u1 = _U_GRID[idx], _U_GRID[idx + 1]
        t = np.clip((u - u0) / (u1 - u0), 0.0, 1.0)
        r = np.arange(len(u))
        out = cols[r, idx] * (1.0 - t) + cols[r, idx + 1] * t
        return np.clip(out, 0.0, None), (dec < lo) | (dec > hi)

    return interp


def declination_coverage(cfg) -> pd.DataFrame:
    """Declination of each configured trace date, to expose sampling gaps."""
    rows = []
    for date in cfg.sweep.dates:
        dec, _ = solar.declination_hour_angle(
            cfg.site.latitude,
            cfg.site.longitude,
            cfg.site.timezone,
            date.year,
            date.month,
            date.day,
            12.0,
        )
        rows.append({"date": date, "declination_deg": dec})
    df = pd.DataFrame(rows).sort_values("declination_deg").reset_index(drop=True)
    df["gap_to_next_deg"] = df["declination_deg"].diff().shift(-1)
    return df


def _declination_of(cfg, date: _dt.date) -> float:
    return solar.declination_hour_angle(
        cfg.site.latitude,
        cfg.site.longitude,
        cfg.site.timezone,
        date.year,
        date.month,
        date.day,
        12.0,
    )[0]


def suggest_sweep_dates(
    cfg,
    n_declinations: int = 8,
    year: int | None = None,
    must_include: tuple[_dt.date, ...] | None = None,
    branch: str = "ascending",
    merge_tolerance_deg: float = 2.0,
) -> list[_dt.date]:
    """Choose trace dates that sample solar declination evenly.

    Traced days reach the annual integral *only* through their declination.
    Sun direction at a fixed site is exactly determined by (declination, hour
    angle), so two dates with the same declination -- one on each side of a
    solstice -- produce identical optics. Only DNI differs between them, and DNI
    is applied per real day at analysis time.

    ``branch``
        ``"ascending"`` samples a single half-year (December solstice to June
        solstice), which sweeps the full declination range without tracing any
        sun direction twice. This is the efficient choice.
        ``"both"`` allows dates from the whole year, which is only useful if you
        want a direct empirical check that the two branches really do agree.

    ``must_include`` dates are always kept (useful when a report needs specific
    calendar dates as figures), but they are not counted twice toward
    declination coverage when they duplicate one another.
    """
    year = year or cfg.sweep.dates[0].year
    must_include = tuple(must_include if must_include is not None else ())

    days = [_dt.date(year, 1, 1) + _dt.timedelta(days=i) for i in range(_days_in(year))]
    decs = np.array([_declination_of(cfg, d) for d in days])

    if branch == "ascending":
        # December solstice -> June solstice: declination rises monotonically.
        lo = int(np.argmin(decs))
        hi = int(np.argmax(decs))
        if lo < hi:
            pool = list(range(lo, hi + 1))
        else:  # wraps the new year
            pool = list(range(lo, len(days))) + list(range(0, hi + 1))
    elif branch == "both":
        pool = list(range(len(days)))
    else:
        raise ValueError(f"branch must be 'ascending' or 'both', got {branch!r}")

    pool_decs = decs[pool]
    targets = np.linspace(decs.min(), decs.max(), n_declinations)

    chosen = set(must_include)
    covered = [_declination_of(cfg, d) for d in chosen]

    for target in targets:
        if covered and np.min(np.abs(np.array(covered) - target)) < merge_tolerance_deg:
            continue
        order = np.argsort(np.abs(pool_decs - target))
        for k in order:
            day = days[pool[int(k)]]
            if day not in chosen:
                chosen.add(day)
                covered.append(pool_decs[int(k)])
                break

    return sorted(chosen)


def distinct_declinations(cfg, dates, tolerance_deg: float = 2.0) -> int:
    """How many *usable* declination samples a date set provides.

    Dates whose declinations fall within ``tolerance_deg`` of each other are
    optically near-duplicates and counted once.
    """
    decs = sorted(_declination_of(cfg, d) for d in dates)
    n = 0
    last = None
    for d in decs:
        if last is None or abs(d - last) >= tolerance_deg:
            n += 1
            last = d
    return n


def _days_in(year: int) -> int:
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def annual_energy(
    summary: pd.DataFrame,
    cfg,
    dni_provider,
    year: int | None = None,
    n_heliostats: int | None = None,
    min_elevation_deg: float = 0.0,
) -> dict:
    """Integrate collected energy over a full year.

    Returns totals plus the hourly and daily breakdowns, so the result can be
    inspected rather than taken on faith.
    """
    year = year or cfg.sweep.dates[0].year
    n_heliostats = n_heliostats or summary["heliostat_id"].nunique()
    mirror_area_total = cfg.field.mirror_area_m2 * n_heliostats

    eff = optical_efficiency(summary, cfg, n_heliostats)
    interp = build_interpolator(eff, cfg.site.latitude)

    hours = solar.hours_of_year(cfg, year)
    daylight = hours["solar_el_deg"] > min_elevation_deg
    hours["eta_optical"] = 0.0
    hours["extrapolated"] = False

    eta, gap = interp(
        hours.loc[daylight, "declination_deg"].to_numpy(),
        hours.loc[daylight, "hour_angle_deg"].to_numpy(),
    )
    hours.loc[daylight, "eta_optical"] = np.clip(eta, 0.0, None)
    hours.loc[daylight, "extrapolated"] = gap

    hours["dni_w_m2"] = [
        dni_provider.dni(d, h) if ok else 0.0
        for d, h, ok in zip(hours["date"], hours["hour"], daylight)
    ]
    # 1 hour timestep -> W becomes Wh directly
    hours["power_w"] = hours["eta_optical"] * hours["dni_w_m2"] * mirror_area_total
    hours["energy_wh"] = hours["power_w"] * 1.0

    daily = hours.groupby("date", as_index=False).agg(
        energy_kwh=("energy_wh", lambda s: s.sum() / 1000.0),
        peak_power_kw=("power_w", lambda s: s.max() / 1000.0),
        dni_kwh_m2=("dni_w_m2", lambda s: s.sum() / 1000.0),
    )

    total_kwh = hours["energy_wh"].sum() / 1000.0
    dni_kwh = hours["dni_w_m2"].sum() / 1000.0
    daylight_hours = hours[daylight]

    return {
        "annual_energy_kwh": total_kwh,
        "annual_energy_mwh": total_kwh / 1000.0,
        "annual_dni_kwh_m2": dni_kwh,
        "mirror_area_m2": mirror_area_total,
        "n_heliostats": n_heliostats,
        "annual_optical_efficiency": (
            total_kwh / (dni_kwh * mirror_area_total) if dni_kwh > 0 else float("nan")
        ),
        "capacity_factor_peak_kw": hours["power_w"].max() / 1000.0,
        "extrapolated_fraction": (
            float(daylight_hours["extrapolated"].mean()) if len(daylight_hours) else float("nan")
        ),
        "traced_timesteps": len(eff),
        "traced_declinations": int(eff["declination_deg"].round(2).nunique()),
        "hourly": hours,
        "daily": daily,
        "efficiency_samples": eff,
    }


def _as_date(value) -> _dt.date:
    return value if isinstance(value, _dt.date) else pd.to_datetime(value).date()


def traced_day_energy(summary: pd.DataFrame, cfg, dni_provider, date, source_date=None) -> dict:
    """Energy collected on one *traced* day, integrated directly over its timesteps.

    This is a second, independent route to a day's energy, deliberately not
    going through :func:`build_interpolator`. Where :func:`annual_energy`
    evaluates an eta_optical(declination, hour_angle) surface on the fixed
    24-sample grid of :func:`solar.hours_of_year`, this multiplies the field
    power actually measured at each traced timestep by the real DNI at that
    date and hour, and integrates over clock time with the trapezoid rule.
    Agreement between the two routes on a traced date is real evidence the
    trace's time sampling is dense enough; a gap is evidence it is not.

    The samples themselves only cover ``sunrise + margin`` .. ``sunset -
    margin``, further narrowed by ``cfg.sweep.min_elevation_deg`` if set (see
    ``solar.build_time_grid``), so the integral is anchored to zero power at
    :func:`solar.elevation_floor_edges` -- the true sunrise/sunset from
    :func:`solar.sunrise_sunset`, narrowed by that same elevation floor.
    Anchoring at the true horizon regardless of the floor would be wrong:
    DNI and the cosine incidence factor really are ~0 exactly at the
    horizon, but a straight line from that true zero to the first sample
    the floor kept (already well above the floor, not just above the
    horizon) cuts across a real, non-linear power ramp and can OVERSTATE the
    day's energy -- measured to happen (a 5 deg floor scoring a higher total
    than a 0 deg one) before this anchor was changed to track the floor.
    Anchoring at the floor's own edge instead makes the anchor coincide with
    the first/last surviving sample (up to the small ``sunrise_margin_min``
    sliver, the one case where the wing is thin enough that a straight line
    across it is a fair approximation, exactly as it always was); the
    excluded band is simply dropped rather than approximated, which only
    ever removes energy from the total, never adds it.

    Deliberately makes no assumption about how many timesteps a day has or
    where they fall in the hour -- the trace grid is not whole-hour and not a
    fixed count.

    ``source_date`` mirrors one traced day onto another calendar day
    -----------------------------------------------------------------
    A trace with a handful of dates relies on the other months of the year
    repeating those declinations closely (to within a fraction of a degree).
    Pass ``source_date`` to take the OPTICS from that traced day while taking
    the DNI and the sunrise/sunset anchors from ``date``: the field's optical
    efficiency depends on where the sun is, which the matched declination
    reproduces, whereas the irradiance and the day length belong to the
    calendar day being reported.

    The result is still built only from traced samples -- no interpolated
    efficiency surface anywhere -- which is the point: a year-scale figure
    that a reader can trace back to real ray-traced instants.

    Default ``None`` means ``source_date = date``, i.e. exactly the behaviour
    this function has always had.
    """
    date = _as_date(date)
    source = date if source_date is None else _as_date(source_date)
    day = summary[summary["date"].apply(_as_date) == source]
    if not len(day):
        raise ValueError(f"No traced timesteps for {source} in this summary")

    grouped = day.groupby("hour", as_index=False)["power_w"].sum().sort_values("hour")
    hours = grouped["hour"].to_numpy(float)
    # power_w is stored at STANDARD_DNI; dni_provider.scale is dni/1000, so this
    # is what actually converts the trace to the real DNI at this date and hour.
    scale = np.array([dni_provider.scale(date, float(h)) for h in hours])
    power_w = grouped["power_w"].to_numpy(float) * scale

    rise, set_ = solar.elevation_floor_edges(cfg, date)

    # Anchor only OUTSIDE the samples. On a day traced against its own date
    # this is unconditional -- build_time_grid samples inside
    # sunrise+margin..sunset-margin -- but a mirrored day carries the source
    # day's hours against this day's horizon, and the two differ by about a
    # minute. An anchor that landed inside the samples would sort into the
    # middle of the array and carve a spurious notch to zero power.
    lead = [rise] if rise < hours[0] else []
    tail = [set_] if set_ > hours[-1] else []
    hours_anchored = np.concatenate((lead, hours, tail))
    power_anchored = np.concatenate((np.zeros(len(lead)), power_w, np.zeros(len(tail))))

    energy_wh = float(np.trapz(power_anchored, hours_anchored))  # W * h = Wh

    return {
        "date": date,
        "source_date": source,
        "energy_kwh": energy_wh / 1000.0,
        "peak_power_kw": float(power_w.max() / 1000.0) if power_w.size else 0.0,
        "sunrise_hour": float(rise),
        "sunset_hour": float(set_),
        "hours": hours,
        "power_kw": power_w / 1000.0,
        "hours_anchored": hours_anchored,
        "power_kw_anchored": power_anchored / 1000.0,
    }


def cross_check_daily_energy(
    summary: pd.DataFrame,
    cfg,
    dni_provider,
    annual: dict | None = None,
    year: int | None = None,
    n_heliostats: int | None = None,
) -> pd.DataFrame:
    """Compare :func:`traced_day_energy` against :func:`annual_energy` on every traced date.

    The two routes share no code path after ``power_w`` -- one interpolates an
    efficiency surface onto 24 fixed hourly samples, the other trapezoids the
    real traced samples with real sunrise/sunset anchors -- so this is a
    genuine consistency check, not a tautology. ``annual`` can be passed in
    (e.g. already computed and cached by a caller) to avoid repeating the
    8760-hour walk.
    """
    if annual is None:
        annual = annual_energy(summary, cfg, dni_provider, year=year, n_heliostats=n_heliostats)
    daily = annual["daily"]
    daily_by_date = {_as_date(d): float(e) for d, e in zip(daily["date"], daily["energy_kwh"])}

    dates = sorted({_as_date(d) for d in summary["date"].unique()})
    rows = []
    for date in dates:
        traced = traced_day_energy(summary, cfg, dni_provider, date)
        interp_kwh = daily_by_date.get(date)
        if interp_kwh is None:
            continue  # traced date falls outside the annual grid's year
        residual_kwh = traced["energy_kwh"] - interp_kwh
        rows.append(
            {
                "date": date,
                "traced_energy_kwh": traced["energy_kwh"],
                "interpolated_energy_kwh": interp_kwh,
                "residual_kwh": residual_kwh,
                "residual_frac": residual_kwh / interp_kwh if interp_kwh else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_annual_sine(daily: pd.DataFrame, value_col: str = "energy_kwh") -> dict:
    """Fit E(d) = A + C*sin(2*pi*d/365.25) + D*cos(2*pi*d/365.25) by linear least squares.

    ``d`` is day-of-year. The model is linear in (A, C, D) once expanded, so
    ``np.linalg.lstsq`` solves it exactly in one shot -- no initial guess, no
    convergence to fail, unlike ``scipy.optimize.curve_fit`` on the same model
    written in amplitude/phase form.

    Returned ``amplitude`` is ``hypot(C, D)`` and ``phase_day_of_year`` is the
    day-of-year the fit peaks at, both derived from writing the fit as
    ``A + amplitude * cos(w*d - phase)``.

    Whether a sinusoid is actually a good description of the site is a
    separate question from whether the fit succeeded -- it always succeeds.
    ``r_squared`` is the honest answer to the first question. At a monsoonal
    site (wet Nov-Mar, dry May-Sep), cloud seasonality can dominate the
    smooth decline-and-recovery a bare sinusoid describes, and a low R² here
    is a real statement about the site's weather, not a bug in the fit.
    """
    doy = pd.to_datetime(daily["date"]).dt.dayofyear.to_numpy(float)
    y = daily[value_col].to_numpy(float)
    w = 2.0 * np.pi / 365.25

    design = np.column_stack([np.ones_like(doy), np.sin(w * doy), np.cos(w * doy)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    A, C, D = (float(v) for v in coeffs)

    y_hat = design @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    amplitude = float(np.hypot(C, D))
    phase_day = float(np.mod(np.arctan2(C, D) / w, 365.25))

    def predict(days):
        d = np.asarray(days, float)
        return A + C * np.sin(w * d) + D * np.cos(w * d)

    return {
        "A": A,
        "C": C,
        "D": D,
        "mean": A,
        "amplitude": amplitude,
        "phase_day_of_year": phase_day,
        "r_squared": r_squared,
        "angular_frequency": w,
        "value_col": value_col,
        "predict": predict,
    }


def per_heliostat_annual(
    summary: pd.DataFrame,
    cfg,
    dni_provider,
    year: int | None = None,
) -> pd.DataFrame:
    """Annual energy attributed to each heliostat.

    Same decomposition as :func:`annual_energy`, applied one heliostat at a
    time. This is what ranks heliostats over a year rather than at one instant --
    a heliostat can look fine at noon and be badly blocked morning and evening.
    """
    year = year or cfg.sweep.dates[0].year
    hours = solar.hours_of_year(cfg, year)
    daylight = hours["solar_el_deg"] > 0.0
    dec_h = hours.loc[daylight, "declination_deg"].to_numpy()
    ha_h = hours.loc[daylight, "hour_angle_deg"].to_numpy()
    dni_h = np.array(
        [
            dni_provider.dni(d, h)
            for d, h in zip(hours.loc[daylight, "date"], hours.loc[daylight, "hour"])
        ]
    )

    results = []
    for hid, grp in summary.groupby("heliostat_id"):
        eff = optical_efficiency(grp, cfg, n_heliostats=1)
        if len(eff) < 3:
            continue
        interp = build_interpolator(eff, cfg.site.latitude)
        eta, _ = interp(dec_h, ha_h)
        eta = np.clip(eta, 0.0, None)
        energy_kwh = float((eta * dni_h * cfg.field.mirror_area_m2).sum() / 1000.0)
        results.append(
            {
                "heliostat_id": hid,
                "x_m": grp["x_m"].iloc[0],
                "y_m": grp["y_m"].iloc[0],
                "radius_m": grp["radius_m"].iloc[0],
                "annual_energy_kwh": energy_kwh,
                "mean_eta_optical": float(np.mean(eta)),
                "min_eta_optical": float(np.min(eta)),
            }
        )

    df = pd.DataFrame(results).sort_values("annual_energy_kwh").reset_index(drop=True)
    if len(df):
        df["rank"] = np.arange(1, len(df) + 1)
        df["pct_of_best"] = 100.0 * df["annual_energy_kwh"] / df["annual_energy_kwh"].max()
    return df
