"""Tests for heliostat.energy.

``tests/fixtures/energy/`` holds a real 10-heliostat, 3-date pytrace run's
``manifest.json`` and ``summary.csv``, plus ``pvgis_tmy.csv`` (a PVGIS TMY
DNI table) and ``expected.json`` recording two full-precision annual kWh
numbers computed from them by the private predecessor of this module: one
against a constant 1000 W/m^2 DNI, one against the PVGIS table read through
the solar-time-alignment wrapper (the table's data longitude, -40.5 deg,
deliberately differs from the site's, -52 deg, so the alignment path is
exercised). Reproducing both to ``rtol=1e-9`` is the gate: it says this port
reads a stored run and integrates its annual energy exactly the way the
private module did.

``cfg`` here is a duck-typed stand-in exposing only what
:mod:`heliostat.energy` reads: ``cfg.site.{latitude,longitude,timezone}`` and
``cfg.field.mirror_area_m2``. Neither field requires the config-loading
machinery this project may grow later.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from heliostat import dni, energy
from heliostat.store import RunStore

FIXTURES = Path(__file__).parent / "fixtures" / "energy"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((FIXTURES / "expected.json").read_text())


@pytest.fixture(scope="module")
def summary(manifest) -> pd.DataFrame:
    # manifest.json + summary.csv are enough for the energy read path: energy
    # consumes a plain summary DataFrame and a duck-typed cfg, never the
    # store's raw/flux files. Going through RunStore.summary() rather than a
    # bare pd.read_csv exercises the same date parsing a real caller gets.
    return RunStore(FIXTURES).summary()


@pytest.fixture(scope="module")
def cfg(manifest, expected):
    return SimpleNamespace(
        site=SimpleNamespace(**manifest["site"]),
        field=SimpleNamespace(mirror_area_m2=expected["mirror_area_m2"]),
    )


@pytest.fixture(scope="module")
def year(expected) -> int:
    return int(expected["dates"][0].split("-")[0])


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------


class TestAnnualEnergyGate:
    def test_constant_dni_matches_expected(self, summary, cfg, expected, year):
        setting = expected["dni_settings"]["constant"]
        assert setting["mode"] == "constant"
        provider = dni.ConstantDNI(setting["value_w_m2"])

        result = energy.annual_energy(
            summary, cfg, provider, year=year, n_heliostats=expected["n_heliostats"]
        )

        assert result["annual_energy_kwh"] == pytest.approx(
            expected["annual_energy_kwh"]["constant_1000_w_m2"], rel=1e-9
        )
        assert result["annual_energy_mwh"] == pytest.approx(
            expected["annual_energy_mwh"]["constant_1000_w_m2"], rel=1e-9
        )

    def test_pvgis_table_solar_time_aligned_matches_expected(self, summary, cfg, expected, year):
        setting = expected["dni_settings"]["table"]
        assert setting["mode"] == "table"
        assert setting["solar_time_aligned"] is True

        frame = pd.read_csv(FIXTURES / "pvgis_tmy.csv")
        base = dni.TableDNI(frame, default=1000.0, source=setting["table_file"])
        provider = dni.SolarTimeAligned(
            base, lon_data_deg=setting["data_longitude"], lon_site_deg=cfg.site.longitude
        )
        # describe() round-trips the exact alignment the fixture was built
        # with -- a mismatch here means the wrong shift would be applied
        # before the number is even compared.
        assert provider.describe() == setting["describe"]

        result = energy.annual_energy(
            summary, cfg, provider, year=year, n_heliostats=expected["n_heliostats"]
        )

        assert result["annual_energy_kwh"] == pytest.approx(
            expected["annual_energy_kwh"]["pvgis_tmy_table"], rel=1e-9
        )
        assert result["annual_energy_mwh"] == pytest.approx(
            expected["annual_energy_mwh"]["pvgis_tmy_table"], rel=1e-9
        )

    def test_traced_sample_counts_match_expected(self, summary, cfg, expected, year):
        # Coarse cross-check on the shape of the traced sample set, not just
        # the final integral -- catches a fixture read that silently drops
        # or duplicates rows while still landing near the right total.
        provider = dni.ConstantDNI(1000.0)
        result = energy.annual_energy(
            summary, cfg, provider, year=year, n_heliostats=expected["n_heliostats"]
        )
        assert result["traced_timesteps"] == expected["traced_timesteps"]
        assert result["traced_declinations"] == expected["traced_declinations"]


# ---------------------------------------------------------------------------
# Clear-sky smoothness (build_interpolator docstring property)
# ---------------------------------------------------------------------------


class TestClearSkySmoothness:
    """No solstice step.

    build_interpolator's docstring records that the old convex-hull form put
    a 3.3-3.7 MWh (3300-3700 kWh) day-to-day jump in the annual total at both
    solstices, because both solstice afternoons fell outside the triangulated
    hull and interpolation silently switched to nearest-neighbour. Clear-sky
    DNI is analytic and the daily curve it drives must be smooth, so a jump
    anywhere near that size at either solstice is exactly the artefact the
    normalised-hour-angle/mirrored-endpoint construction was built to remove.

    The bound here (5 kWh) is generous against genuine day-to-day change --
    the true diurnal curve varies by a fraction of a kWh per day even at the
    solstices (declination is nearly stationary there) -- while still being
    three orders of magnitude below the old bug.
    """

    def test_no_step_at_june_solstice(self, summary, cfg, year):
        self._assert_smooth_around(summary, cfg, year, _dt.date(year, 6, 21))

    def test_no_step_at_december_solstice(self, summary, cfg, year):
        self._assert_smooth_around(summary, cfg, year, _dt.date(year, 12, 21))

    @staticmethod
    def _assert_smooth_around(summary, cfg, year, solstice, window_days=5, max_jump_kwh=5.0):
        provider = dni.ClearSkyDNI(cfg.site)
        result = energy.annual_energy(summary, cfg, provider, year=year, n_heliostats=10)
        daily = result["daily"].copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        lo = pd.Timestamp(solstice) - pd.Timedelta(days=window_days)
        hi = pd.Timestamp(solstice) + pd.Timedelta(days=window_days)
        window = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
        assert len(window) >= 2 * window_days
        jumps = window["energy_kwh"].diff().dropna().abs()
        assert jumps.max() < max_jump_kwh, (
            f"day-to-day jump {jumps.max():.3f} kWh near {solstice} exceeds "
            f"{max_jump_kwh} kWh -- looks like the old convex-hull artefact"
        )


# ---------------------------------------------------------------------------
# declination_coverage sanity
# ---------------------------------------------------------------------------


class TestDeclinationCoverage:
    def test_reports_one_row_per_date_sorted_by_declination(self, cfg, expected):
        dates = [_dt.date.fromisoformat(d) for d in expected["dates"]]
        cfg_with_sweep = SimpleNamespace(site=cfg.site, sweep=SimpleNamespace(dates=dates))

        coverage = energy.declination_coverage(cfg_with_sweep)

        assert len(coverage) == len(dates)
        assert list(coverage["declination_deg"]) == sorted(coverage["declination_deg"])
        # The March and December dates are near opposite solstices/equinox in
        # this fixture (see manifest.json): declination must span a wide
        # range, not cluster near zero.
        assert coverage["declination_deg"].max() - coverage["declination_deg"].min() > 30.0
        # Last row's forward gap is undefined (nothing after it).
        assert pd.isna(coverage["gap_to_next_deg"].iloc[-1])
