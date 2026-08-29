"""Spec §M.7's site DNI control: physics-regression gates.

Four things this control must never get wrong, in order of how badly a
mistake here would be:

1. Left at its default, every surface (single trace, field trace, day
   sweep, aperture-relevant fields) is BIT-IDENTICAL to what it reported
   before this control existed -- the load-bearing pin. See
   ``heliostat.web.app``'s ``DNISetting`` docstring for why the default is
   ``constant`` at 1000 W/m^2 rather than the rider's literally-stated
   "clear-sky model (default)": a clear-sky default would move every
   existing trace fixture's numbers by the sun elevation, which is exactly
   what this file's ``test_*_default_dni_is_bit_identical_*`` tests catch if
   it ever regresses.
2. Power and flux scale EXACTLY linearly with a constant DNI, on both a
   single trace and a field trace -- the one thing a DNI control is FOR.
3. A day sweep's energy integral scales the same way, timestep by timestep.
4. Concentration (flux / DNI) is DNI-INVARIANT: scaling the assumed sun
   scales the flux and the number it is divided by together, so the ratio
   -- the only physically meaningful "concentration" -- must never move.

The year estimate is deliberately exempt from the bit-identical default:
before this control existed it ALREADY used a real, elevation-weighted
:class:`~heliostat.dni.ClearSkyDNI`, hardcoded (the rider's own complaint).
Its default under this control reproduces exactly that (mode="clearsky" is
the site's own default resolution for the year endpoint specifically) --
see ``test_year_default_dni_reproduces_the_old_hardcoded_clearsky_behaviour``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.dni import ClearSkyDNI  # noqa: E402
from heliostat.web.app import DNISetting, create_app  # noqa: E402

RECT_DESIGN = {"type": "rect", "width_mm": 5000, "height_mm": 3000}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _trace_payload(dni=None, solar_el_deg=45.0, **kw):
    payload = {
        "design": RECT_DESIGN,
        "mode": "ultra_fast",
        "optics": "prime_focus",
        "solar_az_deg": 180.0,
        "solar_el_deg": solar_el_deg,
    }
    if dni is not None:
        payload["dni"] = dni
    payload.update(kw)
    return payload


def _field_payload(dni=None, n=4, **kw):
    payload = _trace_payload(dni=dni, **kw)
    payload["layout"] = {"type": "fermat", "n": n}
    return payload


def _constant(value):
    return {"mode": "constant", "constant_w_m2": value}


# ---------------------------------------------------------------------------
# 1. default is bit-identical


def test_single_trace_default_dni_is_bit_identical_to_no_dni_field(client):
    baseline = client.post("/api/trace", json=_trace_payload()).json()
    explicit = client.post("/api/trace", json=_trace_payload(dni=_constant(1000.0))).json()
    assert baseline["power_w"] == explicit["power_w"]
    assert baseline["peak_flux_kw_m2"] == explicit["peak_flux_kw_m2"]
    assert baseline["mean_flux_kw_m2"] == explicit["mean_flux_kw_m2"]
    assert baseline["dni_w_m2"] == 1000.0
    assert baseline["dni_note"] == "1000 W/m² fixed"


def test_field_trace_default_dni_is_bit_identical_to_no_dni_field(client):
    baseline = client.post("/api/field/trace", json=_field_payload()).json()
    explicit = client.post("/api/field/trace", json=_field_payload(dni=_constant(1000.0))).json()
    assert baseline["power_w"] == explicit["power_w"]
    assert baseline["heliostats"] == explicit["heliostats"]
    assert baseline["dni_w_m2"] == 1000.0


def test_day_sweep_default_dni_is_bit_identical_to_no_dni_field(client):
    def run(dni):
        payload = _trace_payload(dni=dni, hour_step=3.0)
        started = client.post("/api/day/start", json=payload)
        assert started.status_code == 200, started.json()
        job_id = started.json()["job_id"]
        for _ in range(600):
            status = client.get(f"/api/day/status/{job_id}").json()
            if status["state"] != "running":
                break
        assert status["state"] == "done", status
        return client.get(f"/api/day/result/{job_id}").json()

    baseline = run(None)
    explicit = run(_constant(1000.0))
    assert baseline["energy_kwh"] == explicit["energy_kwh"]
    assert [s["power_w"] for s in baseline["steps"]] == [s["power_w"] for s in explicit["steps"]]
    assert all(s["dni_w_m2"] == 1000.0 for s in baseline["steps"])
    assert baseline["dni_note"] == "1000 W/m² fixed"


# ---------------------------------------------------------------------------
# 2. linear scaling: single trace, field trace


@pytest.mark.parametrize("value", [500.0, 800.0, 1300.0])
def test_single_trace_power_and_flux_scale_linearly_with_constant_dni(client, value):
    base = client.post("/api/trace", json=_trace_payload(dni=_constant(1000.0))).json()
    scaled = client.post("/api/trace", json=_trace_payload(dni=_constant(value))).json()
    ratio = value / 1000.0
    assert scaled["power_w"] == pytest.approx(base["power_w"] * ratio, rel=1e-9)
    assert scaled["peak_flux_kw_m2"] == pytest.approx(base["peak_flux_kw_m2"] * ratio, rel=1e-9)
    assert scaled["mean_flux_kw_m2"] == pytest.approx(base["mean_flux_kw_m2"] * ratio, rel=1e-9)
    assert scaled["incident_power_w"] == pytest.approx(base["incident_power_w"] * ratio, rel=1e-9)
    # A scale on the source watts must never move where the light lands.
    assert scaled["rms_radius_mm"] == pytest.approx(base["rms_radius_mm"], rel=1e-9)
    assert scaled["centroid_mm"] == pytest.approx(base["centroid_mm"], rel=1e-9)
    assert scaled["dni_w_m2"] == value


@pytest.mark.parametrize("value", [500.0, 1300.0])
def test_field_trace_power_and_flux_scale_linearly_with_constant_dni(client, value):
    base = client.post("/api/field/trace", json=_field_payload(dni=_constant(1000.0))).json()
    scaled = client.post("/api/field/trace", json=_field_payload(dni=_constant(value))).json()
    ratio = value / 1000.0
    assert scaled["power_w"] == pytest.approx(base["power_w"] * ratio, rel=1e-9)
    assert scaled["peak_flux_kw_m2"] == pytest.approx(base["peak_flux_kw_m2"] * ratio, rel=1e-9)
    # Per-heliostat rows carry the same scale, not just the field total.
    for b_row, s_row in zip(base["heliostats"], scaled["heliostats"]):
        assert s_row["id"] == b_row["id"]
        assert s_row["power_w"] == pytest.approx(b_row["power_w"] * ratio, rel=1e-9)
        # eta (occlusion) is a geometry fact, independent of DNI.
        assert s_row["eta"] == b_row["eta"]


def test_field_trace_start_job_scales_the_same_as_the_synchronous_endpoint(client):
    """The background-job field trace (`/api/field/trace/start`) shares
    `_trace_field_heliostats` with the synchronous endpoint -- confirm the
    DNI scale actually threads through THAT call site too, not just the
    one the tests above exercise."""
    sync = client.post("/api/field/trace", json=_field_payload(dni=_constant(700.0))).json()
    started = client.post(
        "/api/field/trace/start", json=_field_payload(dni=_constant(700.0), workers=1)
    )
    assert started.status_code == 200, started.json()
    job_id = started.json()["job_id"]
    for _ in range(600):
        status = client.get(f"/api/field/trace/status/{job_id}").json()
        if status["state"] != "running":
            break
    assert status["state"] == "done", status
    job = client.get(f"/api/field/trace/result/{job_id}").json()
    assert job["power_w"] == pytest.approx(sync["power_w"], rel=1e-9)
    assert job["dni_w_m2"] == 700.0


# ---------------------------------------------------------------------------
# 3. day-sweep energy integral scales linearly


def test_day_sweep_energy_integral_scales_linearly_with_constant_dni(client):
    def run(value):
        payload = _trace_payload(dni=_constant(value), hour_step=3.0)
        started = client.post("/api/day/start", json=payload)
        job_id = started.json()["job_id"]
        for _ in range(600):
            status = client.get(f"/api/day/status/{job_id}").json()
            if status["state"] != "running":
                break
        assert status["state"] == "done", status
        return client.get(f"/api/day/result/{job_id}").json()

    base = run(1000.0)
    scaled = run(400.0)
    # energy_kwh is rounded to 3 decimals server-side (a display rounding,
    # not a physics one), so the linear-scaling check tolerates that last
    # digit rather than demanding exact float equality through it.
    assert scaled["energy_kwh"] == pytest.approx(base["energy_kwh"] * 0.4, abs=2e-3)
    for b_step, s_step in zip(base["steps"], scaled["steps"]):
        # power_w is rounded to 4 decimals, dni_w_m2 to 2 -- both display
        # roundings, not physics ones (see the day/api.py response builder).
        assert s_step["power_w"] == pytest.approx(b_step["power_w"] * 0.4, abs=2e-3)
        assert s_step["dni_w_m2"] == pytest.approx(b_step["dni_w_m2"] * 0.4, abs=2e-2)


# ---------------------------------------------------------------------------
# 4. concentration (flux / DNI) is DNI-invariant


def test_concentration_is_invariant_to_the_constant_dni_chosen(client):
    """avg flux / DNI (the aperture's §M.4 "average concentration") must
    come out the same physical number regardless of which DNI the site
    assumes -- flux and its own divisor scale together by construction."""
    low = client.post("/api/trace", json=_trace_payload(dni=_constant(300.0))).json()
    high = client.post("/api/trace", json=_trace_payload(dni=_constant(1400.0))).json()
    conc_low = low["mean_flux_kw_m2"] * 1000.0 / low["dni_w_m2"]
    conc_high = high["mean_flux_kw_m2"] * 1000.0 / high["dni_w_m2"]
    assert conc_low == pytest.approx(conc_high, rel=1e-9)


# ---------------------------------------------------------------------------
# clear-sky mode: elevation-dependent, matches heliostat.dni.ClearSkyDNI


def test_clearsky_mode_varies_with_elevation_and_matches_the_dni_module(client):
    low_el = client.post(
        "/api/trace", json=_trace_payload(dni={"mode": "clearsky"}, solar_el_deg=15.0)
    ).json()
    high_el = client.post(
        "/api/trace", json=_trace_payload(dni={"mode": "clearsky"}, solar_el_deg=80.0)
    ).json()
    assert high_el["dni_w_m2"] > low_el["dni_w_m2"]
    assert low_el["dni_w_m2"] == pytest.approx(ClearSkyDNI.dni_at_elevation(15.0), rel=1e-9)
    assert high_el["dni_w_m2"] == pytest.approx(ClearSkyDNI.dni_at_elevation(80.0), rel=1e-9)
    assert low_el["dni_note"] == "clear-sky model"


def test_clearsky_scale_factor_scales_the_resolved_dni(client):
    plain = client.post(
        "/api/trace", json=_trace_payload(dni={"mode": "clearsky"}, solar_el_deg=50.0)
    ).json()
    scaled = client.post(
        "/api/trace",
        json=_trace_payload(dni={"mode": "clearsky", "clearsky_scale": 0.8}, solar_el_deg=50.0),
    ).json()
    assert scaled["dni_w_m2"] == pytest.approx(plain["dni_w_m2"] * 0.8, rel=1e-9)
    assert scaled["power_w"] == pytest.approx(plain["power_w"] * 0.8, rel=1e-9)
    assert scaled["dni_note"] == "clear-sky model x0.8"


# ---------------------------------------------------------------------------
# year estimate: default reproduces the old hardcoded ClearSkyDNI behaviour;
# an explicit constant changes it, and scales linearly.

_YEAR_SITE = {"latitude_deg": -10.0, "longitude_deg": -52.0, "timezone_h": -3.0, "year": 2026}


def _year_payload(dni=None, **overrides):
    payload = {
        "design": RECT_DESIGN,
        "mode": "ultra_fast",
        "optics": "prime_focus",
        "solar_az_deg": 180.0,
        "solar_el_deg": 45.0,
        "site": dict(_YEAR_SITE),
        "hour_step": 6.0,
        "fast_mode": True,
    }
    if dni is not None:
        payload["dni"] = dni
    payload.update(overrides)
    return payload


def _run_year(client, dni=None, **overrides):
    started = client.post("/api/year/start", json=_year_payload(dni=dni, **overrides))
    assert started.status_code == 200, started.json()
    job_id = started.json()["job_id"]
    for _ in range(1200):
        status = client.get(f"/api/year/status/{job_id}").json()
        if status["state"] != "running":
            break
    assert status["state"] == "done", status
    return client.get(f"/api/year/result/{job_id}").json()


def test_year_default_dni_reproduces_the_old_hardcoded_clearsky_behaviour(client):
    """No `dni` field at all -- what every year estimate ever posted before
    this control existed -- must still resolve to ClearSkyDNI, unchanged,
    since that WAS this endpoint's only behaviour."""
    result = _run_year(client, dni=None)
    assert "ClearSkyDNI" in result["dni_provider"]
    assert result["dni_note"] == "clear-sky model"
    assert result["annual_energy_mwh"] > 0
    assert result["annual_dni_kwh_m2"] > 0


def test_year_constant_dni_changes_the_total_and_scales_linearly(client):
    clearsky = _run_year(client, dni=None)
    flat_1000 = _run_year(client, dni=_constant(1000.0))
    flat_500 = _run_year(client, dni=_constant(500.0))

    # A flat-1000 assumption is NOT the same as the real elevation-weighted
    # clear-sky curve -- the whole reason this control exists.
    assert flat_1000["annual_energy_mwh"] != pytest.approx(clearsky["annual_energy_mwh"], rel=1e-6)
    assert "ConstantDNI" in flat_1000["dni_provider"] or "1000" in flat_1000["dni_provider"]
    assert flat_1000["dni_note"] == "1000 W/m² fixed"

    # Both figures are rounded to 3/1 decimals server-side (display
    # roundings, not physics ones), so linear scaling is checked to that
    # tolerance rather than demanding exact float equality through it.
    assert flat_500["annual_energy_mwh"] == pytest.approx(flat_1000["annual_energy_mwh"] * 0.5, abs=2e-3)
    assert flat_500["annual_dni_kwh_m2"] == pytest.approx(flat_1000["annual_dni_kwh_m2"] * 0.5, abs=0.2)


# ---------------------------------------------------------------------------
# schema / persistence shape (no HTTP needed)


def test_dni_setting_defaults_and_describe():
    default = DNISetting()
    assert default.mode == "constant"
    assert default.constant_w_m2 == 1000.0
    assert default.describe() == "1000 W/m² fixed"

    custom = DNISetting(mode="constant", constant_w_m2=850.0)
    assert custom.describe() == "850 W/m² fixed"

    clearsky = DNISetting(mode="clearsky")
    assert clearsky.describe() == "clear-sky model"

    clearsky_scaled = DNISetting(mode="clearsky", clearsky_scale=0.9)
    assert clearsky_scaled.describe() == "clear-sky model x0.9"


def test_dni_setting_dni_at_elevation_matches_constant_and_clearsky():
    constant = DNISetting(mode="constant", constant_w_m2=777.0)
    assert constant.dni_at_elevation(10.0) == 777.0
    assert constant.dni_at_elevation(80.0) == 777.0

    clearsky = DNISetting(mode="clearsky")
    assert clearsky.dni_at_elevation(45.0) == pytest.approx(ClearSkyDNI.dni_at_elevation(45.0), rel=1e-12)


def test_project_document_dni_round_trips_through_project_sun():
    from heliostat.web.app import ProjectDocument

    doc = {
        "schema_version": 2,
        "design": RECT_DESIGN,
        "receiver": {"optics": "prime_focus", "params": {}},
        "field": {"layout": None, "heliostat_x_mm": 0.0, "heliostat_y_mm": -89609.0},
        "sun": {
            "azimuth_deg": 180.0,
            "elevation_deg": 45.0,
            "dni": {"mode": "constant", "constant_w_m2": 850.0},
        },
    }
    parsed = ProjectDocument.model_validate(doc)
    assert parsed.sun.dni.mode == "constant"
    assert parsed.sun.dni.constant_w_m2 == 850.0

    # A v1/v2 document saved before §M.7 has no `dni` block at all -- it
    # must still validate, defaulting to constant/1000 so it reopens at
    # exactly the DNI it was always (implicitly) traced at.
    doc["sun"].pop("dni")
    parsed_old = ProjectDocument.model_validate(doc)
    assert parsed_old.sun.dni.mode == "constant"
    assert parsed_old.sun.dni.constant_w_m2 == 1000.0
