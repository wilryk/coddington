"""Smoke tests for ``examples/paper``.

Deliberately tiny: six heliostats, one timestep, the cone backend. A real
reproduction of the paper is hours of Monte Carlo and has no business in a
test suite. What is checked here is that the example's *machinery* works --
that the shipped data files parse and describe the shipped field, that the
config grammar behaves, that a run produces finite, physically sane numbers,
and above all that the ``spherical`` path genuinely reads the frozen-figure
CSV rather than silently falling through to something else.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PAPER_DIR = Path(__file__).resolve().parent.parent / "examples" / "paper"


def _load_reproduce():
    """Import ``examples/paper/reproduce.py`` by path.

    ``examples/`` is not a package and is not installed, so there is no
    import name to use. Loading by path is the honest way to test a script
    that ships as a script.
    """
    spec = importlib.util.spec_from_file_location("paper_reproduce", PAPER_DIR / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def paper():
    return _load_reproduce()


@pytest.fixture(scope="module")
def check_mod():
    spec = importlib.util.spec_from_file_location("paper_check", PAPER_DIR / "check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_field(paper):
    """The six innermost heliostats of the real field."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the coincident-duplicate notice
        return paper.load_paper_field(6)


# ---------------------------------------------------------------------------
# Shipped tables
# ---------------------------------------------------------------------------


def test_expected_tables_have_nine_rows_each(paper):
    annual = pd.read_csv(PAPER_DIR / "expected" / "annual_energy_720mm.csv")
    instant = pd.read_csv(PAPER_DIR / "expected" / "instant_summary.csv")

    assert len(annual) == 9
    assert len(instant) == 9
    for frame in (annual, instant):
        assert set(frame["layout"]) == set(paper.LAYOUTS)
        assert set(frame["figure"]) == set(paper.FIGURES)
        # every (layout, figure) pair present exactly once
        assert len(frame.drop_duplicates(["layout", "figure"])) == 9
    assert {"annual_MWh_1kW", "annual_MWh_petrolina"} <= set(annual.columns)
    assert {"peak_kw_m2", "window_kw", "power_720mm_kw", "frac_720mm"} <= set(instant.columns)
    assert (annual[["annual_MWh_1kW", "annual_MWh_petrolina"]] > 0).all().all()
    # The published numbers are self-consistent: the 720 mm fraction is the
    # ratio of the two power columns, and the concentration is that power over
    # the aperture area in suns.
    assert np.allclose(
        instant["power_720mm_kw"] / instant["window_kw"], instant["frac_720mm"], atol=5e-4
    )
    area = np.pi * (paper.APERTURE_RADIUS_MM / 1000.0) ** 2
    assert np.allclose(instant["power_720mm_kw"] / area, instant["conc_720mm_suns"], rtol=2e-3)


def test_field_file_loads_to_643_heliostats(paper):
    with pytest.warns(UserWarning, match="coincident"):
        field = paper.load_paper_field()
    assert len(field) == paper.PAPER_N_HELIOSTATS == 643
    assert field.dropped_ids == (192, 289)
    radius_m = field.radius_mm / 1000.0
    assert 29.9 < radius_m.min() < 30.1
    assert 89.5 < radius_m.max() < 89.7


@pytest.mark.parametrize("layout", ["prime_focus", "cassegrain", "axicon"])
def test_fixed_shape_tables_describe_the_field(paper, layout):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        field = paper.load_paper_field()
    coeffs = paper.load_fixed_shapes(paper.DATA_DIR / paper.FIXED_SHAPE_FILES[layout], field)
    assert coeffs.shape == (643, 3)
    assert np.isfinite(coeffs).all()
    # Every shipped table is a pure defocus: astigmatic terms are exactly
    # zero, and the defocus term is negative in the trace's sign convention.
    assert (coeffs[:, 0] == 0).all()
    assert (coeffs[:, 2] == 0).all()
    assert (coeffs[:, 1] < 0).all()


def test_fixed_shape_lookup_rejects_a_field_it_does_not_describe(paper, tiny_field):
    """A tolerance that cannot be silently widened by a mismatched field."""
    shifted = type(tiny_field)(
        x_mm=tiny_field.x_mm + 5_000.0,
        y_mm=tiny_field.y_mm + 5_000.0,
        ids=tiny_field.ids,
        mirror_width_mm=tiny_field.mirror_width_mm,
        mirror_height_mm=tiny_field.mirror_height_mm,
    )
    with pytest.raises(ValueError, match="does not describe this field"):
        paper.load_fixed_shapes(paper.DATA_DIR / paper.FIXED_SHAPE_FILES["axicon"], shifted)


# ---------------------------------------------------------------------------
# Config grammar
# ---------------------------------------------------------------------------


def test_parse_configs_default_is_all_nine(paper):
    assert len(paper.parse_configs(None)) == 9
    assert len(paper.parse_configs(["all"])) == 9
    assert set(paper.parse_configs(["all"])) == {
        (lay, fig) for lay in paper.LAYOUTS for fig in paper.FIGURES
    }


def test_parse_configs_accepts_pairs_layouts_and_figures(paper):
    assert paper.parse_configs(["axicon:twisting"]) == [("axicon", "twisting")]
    assert paper.parse_configs(["axicon"]) == [("axicon", f) for f in paper.FIGURES]
    assert paper.parse_configs(["flat"]) == [(lay, "flat") for lay in paper.LAYOUTS]
    # comma-joined, and de-duplicated in first-seen order
    assert paper.parse_configs(["axicon:flat,axicon:flat,cassegrain:flat"]) == [
        ("axicon", "flat"),
        ("cassegrain", "flat"),
    ]


@pytest.mark.parametrize("bad", ["nonsense", "axicon:wobbly", "pyramid:flat"])
def test_parse_configs_rejects_nonsense(paper, bad):
    with pytest.raises(ValueError):
        paper.parse_configs([bad])


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------


def test_shading_body_matches_each_layouts_documented_radius(paper):
    assert paper.paper_optics("prime_focus").shade_body is None
    cas = paper.paper_optics("cassegrain")
    assert cas.shade_body.aperture_radius_mm == 15000.0
    assert cas.secondary.aperture_radius_mm == 14000.0  # the traced surface
    axi = paper.paper_optics("axicon")
    assert axi.shade_body.aperture_radius_mm == 14000.0
    assert paper.paper_optics("axicon", 0).shade_body is None


def test_paper_time_grid_is_94_steps_and_contains_the_instant(paper):
    cfg = paper.paper_cfg(paper.paper_optics("axicon"), n_rays=1)
    from heliostat.solar import build_time_grid

    steps = build_time_grid(cfg, paper.DATES)
    assert len(steps) == 94
    assert paper.INSTANT_KEY in {s.key for s in steps}


# ---------------------------------------------------------------------------
# A tiny end-to-end run
# ---------------------------------------------------------------------------


def _tiny_run(paper, figure, out_dir, field, data_dir=None, monkeypatch=None):
    if data_dir is not None:
        monkeypatch.setattr(paper, "DATA_DIR", data_dir)
    store = paper.run_config(
        "axicon",
        figure,
        out_dir=out_dir,
        dates=paper.DATES[:1],
        mode="ultra_fast",
        n_rays=0,
        field=field,
        only_keys=(paper.INSTANT_KEY,),
        grid_size=64,
        progress=lambda _msg: None,
    )
    return store, paper.instant_metrics(store, store.cfg)


def test_tiny_twisting_run_is_finite_and_sane(paper, tiny_field, tmp_path):
    store, metrics = _tiny_run(paper, "twisting", tmp_path / "twisting", tiny_field)

    assert store.manifest["layout"] == "axicon"
    assert store.manifest["figure"] == "twisting"
    assert store.manifest["sunshape"] == "buie"
    assert store.manifest["n_heliostats"] == 6

    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["window_kw"] > 0
    assert 0.0 <= metrics["frac_720mm"] <= 1.0
    assert metrics["power_720mm_kw"] <= metrics["window_kw"] + 1e-9
    assert 0.0 < metrics["r90_mm"] <= np.hypot(paper.WINDOW_MM, paper.WINDOW_MM)

    # Six 5x3 m mirrors at 1000 W/m^2 with throughput 0.81 cannot deliver more
    # than 6*15*0.81 kW however good the optics are.
    assert metrics["window_kw"] < 6 * 15 * 0.81

    summary = store.summary()
    assert len(summary) == 6
    assert (summary["power_aperture_w"] <= summary["power_w"] + 1e-6).all()
    assert (summary["eta_occlusion"] > 0).all()
    # A twisting run re-solves the figure, so it must not be all zeros.
    assert not np.allclose(summary[["c3", "c4", "c5"]].to_numpy(), 0.0)


def test_retracing_into_an_existing_run_replaces_it(paper, tiny_field, tmp_path):
    """RunStore appends to summary.csv; a re-trace must not double the rows.

    That append is deliberate upstream -- it is what makes a long sweep
    resumable -- so a driver that re-traces into a directory it already wrote
    has to clear it first. Left unhandled, the second run's summary carries
    every heliostat twice and every annual total doubles, silently.
    """
    out = tmp_path / "twice"
    store_a, metrics_a = _tiny_run(paper, "twisting", out, tiny_field)
    assert len(store_a.summary()) == 6

    store_b, metrics_b = _tiny_run(paper, "twisting", out, tiny_field)
    assert len(store_b.summary()) == 6
    assert metrics_b["window_kw"] == pytest.approx(metrics_a["window_kw"], rel=1e-12)


def test_tiny_spherical_run_really_reads_the_csv(paper, tiny_field, tmp_path, monkeypatch):
    """Zeroing the frozen-figure table must change the answer to the flat one.

    This is the test that would catch the spherical path quietly falling back
    to the solved figure, or to no figure at all: with the CSV's coefficients
    replaced by zeros it must reproduce the ``flat`` run exactly, and with the
    real CSV it must not.
    """
    _, real = _tiny_run(paper, "spherical", tmp_path / "sph", tiny_field)
    _, flat = _tiny_run(paper, "flat", tmp_path / "flat", tiny_field)

    # A real frozen sphere concentrates far better than no figure at all.
    assert real["power_720mm_kw"] > 3.0 * flat["power_720mm_kw"]

    # Now zero the table the spherical path reads, and nothing else.
    fake_data = tmp_path / "fake_data"
    fake_data.mkdir()
    name = paper.FIXED_SHAPE_FILES["axicon"]
    table = pd.read_csv(paper.DATA_DIR / name, comment="#")
    table[["c3", "c4", "c5"]] = 0.0
    table.to_csv(fake_data / name, index=False)
    shutil.copy(paper.DATA_DIR / paper.FIELD_FILE, fake_data / paper.FIELD_FILE)

    _, zeroed = _tiny_run(
        paper, "spherical", tmp_path / "zeroed", tiny_field, fake_data, monkeypatch
    )

    assert zeroed["power_720mm_kw"] == pytest.approx(flat["power_720mm_kw"], rel=1e-12)
    assert zeroed["peak_kw_m2"] == pytest.approx(flat["peak_kw_m2"], rel=1e-12)
    assert real["power_720mm_kw"] != pytest.approx(zeroed["power_720mm_kw"], rel=1e-6)


def test_petrolina_dni_provider_matches_the_published_record(paper):
    """The shipped climatology must integrate to the record it came from."""
    provider = paper.petrolina_provider()
    assert provider.inner.n_years == 24
    assert provider.inner.window_days == 5
    # The paper quotes 1,848.6 kWh/m2/yr for the raw record; the +/-5-day
    # smoothing is mass-preserving to well within a tenth of a percent.
    assert provider.inner.annual_kwh_m2() == pytest.approx(1848.6, rel=2e-3)
    # Read at the site's solar time, not the record's clock: Petrolina is
    # 11.5 deg east of the site, so the table is read 46 minutes early.
    assert provider.offset_h == pytest.approx(-11.5 / 15.0)


# ---------------------------------------------------------------------------
# check.py
# ---------------------------------------------------------------------------


def test_check_loads_the_expected_tables(check_mod):
    expected = check_mod.load_expected()
    assert len(expected) == 9
    assert not expected.isna().any().any()


def test_check_labels_a_non_paper_run_as_not_comparable(check_mod, tmp_path):
    """A quick/tiny run must be reported ``n/c``, never PASS and never FAIL."""
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        [
            {
                "layout": "axicon",
                "figure": "twisting",
                "annual_MWh_1kW": 1.0,  # wildly wrong on purpose
                "annual_MWh_petrolina": 1.0,
                "peak_kw_m2": 1.0,
                "window_kw": 1.0,
                "power_720mm_kw": 1.0,
                "frac_720mm": 0.5,
                "conc_720mm_suns": 1.0,
                "r90_mm": 1.0,
            }
        ]
    ).to_csv(results / "summary.csv", index=False)
    (results / "axicon_twisting.json").write_text(
        '{"layout": "axicon", "figure": "twisting", "mode": "ultra_fast", '
        '"rays_per_heliostat": 0, "n_heliostats": 60, "grid_size": 256, '
        '"traced_dates": ["20261221"], '
        '"paper_comparable": {"instant": false, "annual": false}}'
    )

    table, ok, n_compared = check_mod.compare(tmp_path)
    assert n_compared == 0
    assert ok is True
    assert set(table["verdict"]) == {"n/c"}
    assert check_mod.main(["--out", str(tmp_path)]) == 0
