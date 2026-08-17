"""Tests for heliostat.field_layouts and the ``heliostat layout`` CLI.

Covers the MATLAB-pinned spiral formula, the wedge+road filter combination
against an independent reimplementation of the MATLAB script's logic,
``generate()``'s truncation/starvation behaviour, ``min_spacing_filter``
against a brute-force reference, the CSV round-trip through
:func:`heliostat.field.load_field`, and the CLI end to end.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from heliostat.cli import main
from heliostat.field import load_field
from heliostat.field_layouts import (
    GOLDEN,
    fermat_spiral,
    generate,
    min_spacing_filter,
    ring_filter,
    road_corridors,
    wedge_filter,
    write_field_csv,
)

# ---------------------------------------------------------------------------
# MATLAB pins
# ---------------------------------------------------------------------------


def _matlab_point(a: float, b: float, k: int) -> tuple[float, float]:
    """Hand-computed reference from FermatSpiral.m's formulas, independent of
    fermat_spiral()'s implementation."""
    phi_g = (math.sqrt(5) + 1) / 2
    theta_k = 2 * math.pi / phi_g**2 * k
    r_k = a * k**b
    return r_k * math.cos(theta_k), r_k * math.sin(theta_k)


@pytest.mark.parametrize("k", [100, 1000])
def test_fermat_spiral_matches_matlab_pin(k):
    a, b = 3.6, 0.56
    x_exp, y_exp = _matlab_point(a, b, k)
    (x, y) = fermat_spiral(1, a, b=b, k_start=k)[0]
    assert x == pytest.approx(x_exp, abs=1e-12)
    assert y == pytest.approx(y_exp, abs=1e-12)


def test_golden_constant():
    assert GOLDEN == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-15)


def test_fermat_spiral_k_start_matches_oversample_then_ring_filter():
    """The two ways of reproducing MATLAB's kmin:kmax range (direct
    k_start, or oversample-from-1 + ring_filter) must give identical
    points, since the formulas are pure functions of k."""
    a, b = 3.6, 0.56
    kmin, kmax = 81, 120
    direct = fermat_spiral(kmax - kmin + 1, a, b=b, k_start=kmin)

    wide = fermat_spiral(kmax, a, b=b, k_start=1)
    r_min = a * kmin**b - 1e-9
    r_max = a * kmax**b + 1e-9
    mask = ring_filter(r_min, r_max)(wide)
    filtered = wide[mask]

    np.testing.assert_allclose(filtered, direct, atol=1e-9)


def test_fermat_spiral_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        fermat_spiral(0, 1.0)


# ---------------------------------------------------------------------------
# Wedge + road example vs. an independent MATLAB reimplementation
# ---------------------------------------------------------------------------


def test_wedge_and_road_survivor_count_matches_matlab_reimplementation():
    a, b = 3.6, 0.56
    rmin, rmax = 42, 200
    kmin = math.ceil((rmin / a) ** (1 / b))
    kmax = math.floor((rmax / a) ** (1 / b))
    n = kmax - kmin + 1

    xy = fermat_spiral(n, a, b=b, k_start=kmin)
    mask = wedge_filter(45, 135)(xy)
    mask &= road_corridors(10, azimuths_deg=(180,))(xy)
    got_count = int(mask.sum())

    # Independent reimplementation, straight from FermatSpiral.m.
    phi_g = (math.sqrt(5) + 1) / 2
    k = np.arange(kmin, kmax + 1)
    theta_k = 2 * np.pi / phi_g**2 * k
    r_k = a * k**b
    x = r_k * np.cos(theta_k)
    y = r_k * np.sin(theta_k)
    theta = np.arctan2(y, x)
    filt = (theta >= np.pi / 4) & (theta <= 3 * np.pi / 4)
    xp, yp = x[filt], y[filt]
    survives_road = (xp > 10) | (xp < -10) | (yp >= 0)
    expected_count = int(survives_road.sum())

    assert got_count == expected_count
    assert got_count > 0  # sanity: the example isn't vacuous


def test_wedge_filter_wraparound():
    xy = np.array([[1.0, 0.0], [-1.0, 0.1], [-1.0, -0.1], [0.0, 1.0]])
    # Band straddling +-180 deg: keep near-negative-x-axis points only.
    mask = wedge_filter(170, -170)(xy)
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_ring_filter_bounds_inclusive():
    xy = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    mask = ring_filter(10.0, 20.0)(xy)
    np.testing.assert_array_equal(mask, [True, True, False])


def test_road_corridors_only_blocks_the_named_side():
    # South corridor: y<0 & |x|<=10 removed; y>0 counterpart untouched.
    xy = np.array([[0.0, -5.0], [0.0, 5.0], [15.0, -5.0]])
    mask = road_corridors(10.0, azimuths_deg=(180.0,))(xy)
    np.testing.assert_array_equal(mask, [False, True, True])


# ---------------------------------------------------------------------------
# min_spacing_filter vs brute force
# ---------------------------------------------------------------------------


def _brute_force_min_spacing(xy: np.ndarray, min_m: float) -> np.ndarray:
    n = xy.shape[0]
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        conflict = False
        for j in range(i):
            if mask[j] and np.hypot(*(xy[i] - xy[j])) <= min_m:
                conflict = True
                break
        mask[i] = not conflict
    return mask


def test_min_spacing_filter_matches_brute_force():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 50, size=(150, 2))
    expected = _brute_force_min_spacing(xy, 5.0)
    got = min_spacing_filter(5.0)(xy)
    np.testing.assert_array_equal(got, expected)


def test_min_spacing_filter_keeps_lower_index_on_conflict():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    mask = min_spacing_filter(1.5)(xy)
    # 0 kept; 1 conflicts with 0 (dist 1<=1.5), dropped; 2 vs 0 dist=2>1.5 OK,
    # 2 vs 1 doesn't matter since 1 was dropped.
    np.testing.assert_array_equal(mask, [True, False, True])


def test_min_spacing_filter_empty():
    xy = np.empty((0, 2))
    mask = min_spacing_filter(1.0)(xy)
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_exactly_n_target():
    field = generate("fermat", 200, a_m=4.0, b=0.5)
    assert len(field) == 200
    np.testing.assert_array_equal(field.ids, np.arange(200))


def test_generate_truncates_by_k_order():
    """With no filters, generate() must keep the first n_target points in k
    order (i.e. it truncates, not resamples/reorders)."""
    field = generate("fermat", 50, a_m=4.0, b=0.5, oversample=2.0)
    direct = fermat_spiral(50, 4.0, b=0.5, k_start=1)
    np.testing.assert_allclose(field.x_m, direct[:, 0])
    np.testing.assert_allclose(field.y_m, direct[:, 1])


def test_generate_raises_informatively_when_filters_starve_it():
    with pytest.raises(ValueError, match="only .* of .* requested"):
        generate(
            "fermat",
            5000,
            a_m=4.0,
            b=0.5,
            filters=(ring_filter(0.0, 1.0),),  # almost nothing survives this near the origin
            oversample=1.2,
        )


def test_generate_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown layout kind"):
        generate("nonsense", 10, a_m=1.0)


def test_generate_records_source():
    field = generate("fermat", 20, a_m=4.0, b=0.5)
    assert "fermat" in field.source
    assert "a_m=4.0" in field.source


def test_generate_rejects_nonpositive_n_target():
    with pytest.raises(ValueError):
        generate("fermat", 0, a_m=4.0)


# ---------------------------------------------------------------------------
# CSV round trip
# ---------------------------------------------------------------------------


def test_write_field_csv_round_trip(tmp_path):
    field = generate("fermat", 80, a_m=4.5, b=0.55, filters=(ring_filter(5, 300),))
    path = tmp_path / "field.csv"
    write_field_csv(field, path)

    loaded = load_field(path)
    assert len(loaded) == len(field)
    np.testing.assert_array_equal(loaded.ids, np.arange(len(field)))
    np.testing.assert_allclose(loaded.x_m, field.x_m, atol=1e-9)
    np.testing.assert_allclose(loaded.y_m, field.y_m, atol=1e-9)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_layout_fermat_in_process(tmp_path, capsys):
    out = tmp_path / "field.csv"
    rc = main(
        [
            "layout",
            "fermat",
            "--n",
            "150",
            "--a",
            "4.5",
            "--b",
            "0.55",
            "--ring",
            "20",
            "200",
            "--road-width",
            "6",
            "--road-az",
            "180",
            "--min-spacing",
            "8",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "150 heliostats" in captured.out
    assert "land coverage" in captured.out

    assert out.exists()
    loaded = load_field(out)
    assert len(loaded) == 150


def test_cli_layout_fermat_subprocess(tmp_path):
    out = tmp_path / "field_subproc.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "heliostat.cli",
            "layout",
            "fermat",
            "--n",
            "40",
            "--a",
            "4.0",
            "-o",
            str(out),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "40 heliostats" in result.stdout
    assert out.exists()
    loaded = load_field(out)
    assert len(loaded) == 40


def _subprocess_env():
    import os
    from pathlib import Path

    env = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_serve_and_version_still_registered():
    """Non-destructive extension check: `serve` and `--version` must still
    work after adding `layout`. argparse's --help/--version both raise
    SystemExit(0) rather than returning, so that's what's asserted."""
    with pytest.raises(SystemExit) as exc_info:
        main(["serve", "--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_bare_layout_without_kind_prints_help(capsys):
    rc = main(["layout"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "fermat" in captured.out
