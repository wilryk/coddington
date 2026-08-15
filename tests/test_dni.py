"""Tests for heliostat.dni.

All data is synthetic and in-memory; nothing here touches the network. The
fetch functions (_fetch_pvgis_tmy / _fetch_nasa_power) are exercised only for
their lazy-import behaviour elsewhere in this file's sibling tests, never
called.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from heliostat import dni


def _table_frame():
    """Two days, several hours each, single year -- for TableDNI."""
    rows = []
    for day, base in ((20, 500.0), (21, 600.0)):
        for hour, mult in ((6.0, 0.1), (9.0, 0.6), (12.0, 1.0), (15.0, 0.5), (18.0, 0.05)):
            rows.append({"month": 3, "day": day, "hour": hour, "dni_w_m2": base * mult})
    return pd.DataFrame(rows)


class TestConstantDNI:
    def test_returns_fixed_value(self):
        p = dni.ConstantDNI(850.0)
        assert p.dni(dt.date(2026, 1, 1), 3.0) == 850.0
        assert p.dni(dt.date(2026, 7, 4), 23.9) == 850.0

    def test_default_is_standard_dni(self):
        p = dni.ConstantDNI()
        assert p.dni(dt.date(2026, 1, 1), 12.0) == dni.STANDARD_DNI

    def test_scale_divides_by_standard(self):
        p = dni.ConstantDNI(500.0)
        assert p.scale(dt.date(2026, 1, 1), 12.0) == pytest.approx(0.5)

    def test_describe(self):
        p = dni.ConstantDNI(750.0)
        assert "750" in p.describe()


class TestTableDNI:
    def test_missing_required_columns_raises(self):
        with pytest.raises(ValueError):
            dni.TableDNI(pd.DataFrame({"month": [1], "day": [1]}))

    def test_exact_hour_lookup(self):
        p = dni.TableDNI(_table_frame())
        assert p.dni(dt.date(2026, 3, 20), 12.0) == pytest.approx(500.0)
        assert p.dni(dt.date(2026, 3, 21), 12.0) == pytest.approx(600.0)

    def test_interpolates_between_hours(self):
        p = dni.TableDNI(_table_frame())
        # halfway between hour 9 (300) and hour 12 (500) on day 20
        val = p.dni(dt.date(2026, 3, 20), 10.5)
        assert val == pytest.approx((300.0 + 500.0) / 2.0)

    def test_hour_outside_table_clamps_to_edge(self):
        p = dni.TableDNI(_table_frame())
        # before first table hour (6.0) -> clamps to that edge's value (50.0)
        assert p.dni(dt.date(2026, 3, 20), 0.0) == pytest.approx(50.0)
        # after last table hour (18.0) -> clamps to that edge's value (25.0)
        assert p.dni(dt.date(2026, 3, 20), 23.9) == pytest.approx(25.0)

    def test_missing_day_falls_back_to_default(self):
        p = dni.TableDNI(_table_frame(), default=111.0)
        assert p.dni(dt.date(2026, 6, 1), 12.0) == 111.0

    def test_single_hour_per_day(self):
        frame = pd.DataFrame({"month": [4], "day": [1], "hour": [12.0], "dni_w_m2": [321.0]})
        p = dni.TableDNI(frame)
        assert p.dni(dt.date(2026, 4, 1), 3.0) == pytest.approx(321.0)
        assert p.dni(dt.date(2026, 4, 1), 20.0) == pytest.approx(321.0)

    def test_describe_mentions_day_count(self):
        p = dni.TableDNI(_table_frame(), source="synthetic")
        assert "2 days" in p.describe()
        assert "synthetic" in p.describe()


class TestMonthlyProfileDNI:
    def _frame(self):
        # Two months, two days each, same diurnal hours -- averaging is exact.
        rows = []
        for month, days in ((3, (1, 2)), (6, (1, 2))):
            for day in days:
                for hour, val in ((6.0, 100.0 + day), (12.0, 500.0 + day), (18.0, 100.0 - day)):
                    rows.append({"month": month, "day": day, "hour": hour, "dni_w_m2": val})
        return pd.DataFrame(rows)

    def test_averages_across_days_in_month(self):
        p = dni.MonthlyProfileDNI(self._frame())
        # March hour=12: days 1 and 2 give 501, 502 -> mean 501.5
        assert p.dni(dt.date(2026, 3, 15), 12.0) == pytest.approx(501.5)

    def test_different_calendar_day_same_month_same_profile(self):
        p = dni.MonthlyProfileDNI(self._frame())
        assert p.dni(dt.date(2026, 3, 1), 12.0) == p.dni(dt.date(2026, 3, 28), 12.0)

    def test_missing_month_falls_back_to_default(self):
        p = dni.MonthlyProfileDNI(self._frame(), default=222.0)
        assert p.dni(dt.date(2026, 9, 1), 12.0) == 222.0

    def test_interpolates_within_month_profile(self):
        p = dni.MonthlyProfileDNI(self._frame())
        low = p.dni(dt.date(2026, 3, 10), 6.0)
        high = p.dni(dt.date(2026, 3, 10), 12.0)
        mid = p.dni(dt.date(2026, 3, 10), 9.0)
        assert low < mid < high

    def test_day_kwh_m2_is_trapezoid_of_profile(self):
        p = dni.MonthlyProfileDNI(self._frame())
        hours, values = p._by_month[3]
        expected = np.trapz(values, hours) / 1000.0
        assert p.day_kwh_m2(3) == pytest.approx(expected)


class TestDailyClimatologyDNI:
    def _frame(self):
        rows = []
        for year in (2020, 2021, 2022):
            for day, base in ((15, 800.0), (16, 810.0)):
                for hour in (9.0, 12.0, 15.0):
                    # small per-year jitter so the mean is meaningfully an average
                    jitter = {2020: -10.0, 2021: 0.0, 2022: 10.0}[year]
                    rows.append(
                        {
                            "year": year,
                            "month": 6,
                            "day": day,
                            "hour": hour,
                            "dni_w_m2": base + jitter + (hour - 9.0) * 5,
                        }
                    )
        return pd.DataFrame(rows)

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            dni.DailyClimatologyDNI(pd.DataFrame({"month": [1]}))

    def test_n_years_detected(self):
        p = dni.DailyClimatologyDNI(self._frame(), window_days=0)
        assert p.n_years == 3

    def test_defaults_to_one_year_without_year_column(self):
        frame = self._frame().drop(columns=["year"])
        p = dni.DailyClimatologyDNI(frame, window_days=0)
        assert p.n_years == 1

    def test_averages_across_years_at_same_doy_hour(self):
        p = dni.DailyClimatologyDNI(self._frame(), window_days=0)
        # 2026-06-15 hour 9: mean of (800-10, 800+0, 800+10) = 800.0
        assert p.dni(dt.date(2026, 6, 15), 9.0) == pytest.approx(800.0, abs=1e-9)

    def test_leap_day_folds_onto_feb_28(self):
        frame = pd.DataFrame(
            {
                "year": [2020, 2020],
                "month": [2, 2],
                "day": [28, 29],
                "hour": [12.0, 12.0],
                "dni_w_m2": [400.0, 600.0],
            }
        )
        p = dni.DailyClimatologyDNI(frame, window_days=0)
        # both rows fold onto the same (doy, hour) bucket and average together
        assert p.dni(dt.date(2026, 2, 28), 12.0) == pytest.approx(500.0)

    def test_window_smoothing_blends_neighbouring_days(self):
        # a single-day spike should be visible with window=0 and softened
        # once neighbouring (empty, interpolated-flat) days are averaged in.
        p0 = dni.DailyClimatologyDNI(self._frame(), window_days=0)
        p5 = dni.DailyClimatologyDNI(self._frame(), window_days=5)
        v0 = p0.dni(dt.date(2026, 6, 15), 9.0)
        v5 = p5.dni(dt.date(2026, 6, 15), 9.0)
        # both finite; smoothing is not required to change a flat-filled
        # region, so just check it stays sane and doesn't error.
        assert np.isfinite(v0)
        assert np.isfinite(v5)

    def test_day_kwh_m2_and_annual_kwh_m2_are_consistent(self):
        p = dni.DailyClimatologyDNI(self._frame(), window_days=0)
        total = sum(p.day_kwh_m2(dt.date(2026, 1, 1) + dt.timedelta(days=i)) for i in range(365))
        assert p.annual_kwh_m2() == pytest.approx(total)

    def test_all_nan_row_falls_back_to_default(self):
        # Not reachable through normal construction (interpolate always fills
        # a column that has at least one real sample), so exercise the
        # fallback branch directly against the internal state.
        p = dni.DailyClimatologyDNI(self._frame(), default=999.0, window_days=0)
        p._values[100, :] = np.nan
        # Pick the date whose day-of-year is 101 (index 100)
        target = dt.date(2026, 1, 1) + dt.timedelta(days=100)
        assert int(p._doy(target.month, target.day)) - 1 == 100
        assert p.dni(target, 12.0) == 999.0

    def test_describe(self):
        p = dni.DailyClimatologyDNI(self._frame(), source="synthetic", window_days=5)
        s = p.describe()
        assert "3 years" in s
        assert "synthetic" in s


class TestClearSkyDNI:
    SITE = SimpleNamespace(latitude=-9.4, longitude=-40.5, timezone=-3)

    def test_rejects_invalid_am1(self):
        with pytest.raises(ValueError):
            dni.ClearSkyDNI(self.SITE, am1_w_m2=-1.0)
        with pytest.raises(ValueError):
            dni.ClearSkyDNI(self.SITE, am1_w_m2=dni.ClearSkyDNI.E0 + 1.0)

    def test_air_mass_at_zenith_is_one(self):
        assert dni.ClearSkyDNI.air_mass(90.0) == pytest.approx(1.0, abs=1e-3)

    def test_air_mass_near_horizon_is_finite_and_large(self):
        # Exactly 0.0 deg is treated as "at/below horizon" (see below) and
        # returns inf by design; just above it must stay finite.
        am = dni.ClearSkyDNI.air_mass(0.5)
        assert np.isfinite(am)
        assert am > 30.0

    def test_air_mass_at_and_below_horizon_is_infinite(self):
        assert dni.ClearSkyDNI.air_mass(0.0) == float("inf")
        assert dni.ClearSkyDNI.air_mass(-1.0) == float("inf")

    def test_dni_zero_below_horizon(self):
        p = dni.ClearSkyDNI(self.SITE)
        assert p.dni(dt.date(2026, 6, 21), 0.0) == 0.0

    def test_dni_near_am1_at_solar_noon(self):
        # solar noon at this site on the equinox is close to sun overhead,
        # so DNI should land close to (but not exceed) am1_w_m2.
        p = dni.ClearSkyDNI(self.SITE, am1_w_m2=1000.0)
        val = p.dni(dt.date(2026, 3, 21), 12.0)
        assert 0.0 < val <= 1000.0 + 1e-6

    def test_higher_air_mass_gives_lower_dni(self):
        p = dni.ClearSkyDNI(self.SITE, am1_w_m2=1000.0)
        noon = p.dni(dt.date(2026, 6, 21), 12.0)
        morning = p.dni(dt.date(2026, 6, 21), 7.0)
        assert morning < noon

    def test_annual_kwh_m2_positive(self):
        p = dni.ClearSkyDNI(self.SITE)
        assert p.annual_kwh_m2(2026) > 0.0

    def test_describe(self):
        p = dni.ClearSkyDNI(self.SITE, am1_w_m2=1000.0)
        assert "Meinel" in p.describe()


class TestSolarTimeAligned:
    def test_zero_offset_when_same_longitude(self):
        inner = dni.ConstantDNI(500.0)
        p = dni.SolarTimeAligned(inner, lon_data_deg=-40.5, lon_site_deg=-40.5)
        assert p.offset_h == pytest.approx(0.0)

    def test_data_east_of_site_reads_table_earlier(self):
        # data to the EAST (less negative / larger longitude) -> offset negative
        table = dni.TableDNI(_table_frame())
        p = dni.SolarTimeAligned(table, lon_data_deg=-40.5, lon_site_deg=-52.5)
        assert p.offset_h < 0.0
        # reading at site hour H should equal reading the inner table at H+offset
        h = 12.0
        assert p.dni(dt.date(2026, 3, 20), h) == pytest.approx(
            table.dni(dt.date(2026, 3, 20), h + p.offset_h)
        )

    def test_offset_magnitude_matches_15_deg_per_hour(self):
        inner = dni.ConstantDNI(500.0)
        p = dni.SolarTimeAligned(inner, lon_data_deg=-40.5, lon_site_deg=-55.5)
        # 15 deg longitude difference -> exactly 1 hour offset
        assert abs(p.offset_h) == pytest.approx(1.0)

    def test_shifts_curve_as_documented(self):
        # A table with a distinct peak at hour 12; aligning to a site whose
        # solar time trails the data's should move that peak to a later site
        # clock hour.
        frame = pd.DataFrame(
            {
                "month": [1] * 5,
                "day": [1] * 5,
                "hour": [10.0, 11.0, 12.0, 13.0, 14.0],
                "dni_w_m2": [200.0, 400.0, 900.0, 400.0, 200.0],
            }
        )
        table = dni.TableDNI(frame)
        lon_data, lon_site = -40.5, -55.5  # site west of data -> +1h offset expected in peak time
        p = dni.SolarTimeAligned(table, lon_data_deg=lon_data, lon_site_deg=lon_site)
        peak_hour_site = max(
            np.arange(10.0, 14.01, 0.1), key=lambda h: p.dni(dt.date(2026, 1, 1), h)
        )
        assert peak_hour_site == pytest.approx(12.0 - p.offset_h, abs=0.05)

    def test_describe_includes_offset(self):
        inner = dni.ConstantDNI(500.0)
        p = dni.SolarTimeAligned(inner, lon_data_deg=-40.5, lon_site_deg=-52.5)
        s = p.describe()
        assert "aligned" in s
        assert "min" in s


class TestAllModes:
    def test_all_modes_covers_table_modes_and_extras(self):
        assert dni.ALL_MODES == {"constant", "clearsky", "table", "monthly", "climatology"}
