"""Tests for heliostat.field.

Covers the flexible-column loader on a synthetic file, ``neighbour_pairs``
against a brute-force O(N^2) reference, and the coincident-position
warn+drop behaviour ``load_field`` applies automatically.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from heliostat.field import (
    HeliostatField,
    coincident_pairs,
    downselect,
    load_field,
    neighbour_pairs,
)

# ---------------------------------------------------------------------------
# Loader: flexible column matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "columns",
    [
        ("X (m)", "Y (m)"),
        ("x_s (m)", "y_s (m)"),
        ("x", "y"),
        ("X (MM)", "Y (Mm)"),
    ],
)
def test_load_field_flexible_columns(tmp_path, columns):
    """Odd header spellings/casing all resolve to the same x/y aliases."""
    xcol, ycol = columns
    values_m = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    if "mm" in xcol.lower():
        df = pd.DataFrame({xcol: values_m[:, 0] * 1000.0, ycol: values_m[:, 1] * 1000.0})
    else:
        df = pd.DataFrame({xcol: values_m[:, 0], ycol: values_m[:, 1]})
    path = tmp_path / "field.csv"
    df.to_csv(path, index=False)

    fld = load_field(path, mirror_width_mm=5000.0, mirror_height_mm=3000.0)

    assert len(fld) == 3
    np.testing.assert_allclose(fld.x_m, values_m[:, 0])
    np.testing.assert_allclose(fld.y_m, values_m[:, 1])
    assert fld.mirror_width_mm == 5000.0
    assert fld.mirror_height_mm == 3000.0
    assert fld.dropped_ids == ()


def test_load_field_missing_column_raises(tmp_path):
    df = pd.DataFrame({"foo": [1.0, 2.0], "bar": [3.0, 4.0]})
    path = tmp_path / "field.csv"
    df.to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_field(path)


def test_load_field_xlsx(tmp_path):
    df = pd.DataFrame({"x (m)": [0.0, 5.0], "y (m)": [0.0, 5.0]})
    path = tmp_path / "field.xlsx"
    df.to_excel(path, index=False)
    fld = load_field(path)
    assert len(fld) == 2
    np.testing.assert_allclose(fld.x_m, [0.0, 5.0])


# ---------------------------------------------------------------------------
# neighbour_pairs vs brute force
# ---------------------------------------------------------------------------


def test_neighbour_pairs_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 200
    xy = rng.uniform(-100_000.0, 100_000.0, size=(n, 2))
    fld = HeliostatField(x_mm=xy[:, 0], y_mm=xy[:, 1], ids=np.arange(n))

    radius_mm = 15_000.0
    got = neighbour_pairs(fld, radius_mm)

    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    for i in range(n):
        expected = np.flatnonzero((dist[i] <= radius_mm) & (np.arange(n) != i))
        np.testing.assert_array_equal(np.sort(got[i]), expected)


def test_neighbour_pairs_excludes_self():
    fld = HeliostatField(x_mm=np.array([0.0, 0.0]), y_mm=np.array([0.0, 0.0]), ids=np.array([0, 1]))
    got = neighbour_pairs(fld, 1.0)
    assert list(got[0]) == [1]
    assert list(got[1]) == [0]


# ---------------------------------------------------------------------------
# Coincident-pair detection, warning, and drop
# ---------------------------------------------------------------------------


def test_coincident_pairs_detects_close_positions():
    fld = HeliostatField(
        x_mm=np.array([0.0, 0.0001, 5000.0, 10000.0]),
        y_mm=np.array([0.0, 0.0, 5000.0, 10000.0]),
        ids=np.array([0, 1, 2, 3]),
    )
    pairs = coincident_pairs(fld, tol_mm=1.0)
    assert pairs == [(0, 1)]


def test_load_field_drops_coincident_and_warns(tmp_path):
    """Two rows within 1 mm of each other: the higher id is dropped, the
    caller is warned by name, and the survivor records what was dropped --
    generic detection, not a hardcoded id list."""
    df = pd.DataFrame(
        {
            "x (m)": [0.0, 0.0, 20.0, 30.0],
            "y (m)": [0.0, 0.0000001, 20.0, 30.0],
        }
    )
    path = tmp_path / "dupes.csv"
    df.to_csv(path, index=False)

    with pytest.warns(UserWarning, match="coincident"):
        fld = load_field(path)

    assert len(fld) == 3
    assert fld.dropped_ids == (1,)
    assert list(fld.ids) == [0, 2, 3]


def test_load_field_no_warning_without_duplicates(tmp_path):
    df = pd.DataFrame({"x (m)": [0.0, 20.0, 40.0], "y (m)": [0.0, 20.0, 40.0]})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fld = load_field(path)
    assert fld.dropped_ids == ()
    assert len(fld) == 3


def test_subset_by_id_survives_drop(tmp_path):
    df = pd.DataFrame({"x (m)": [0.0, 0.0, 20.0], "y (m)": [0.0, 0.0, 20.0]})
    path = tmp_path / "dupes.csv"
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning):
        fld = load_field(path)
    sub = fld.subset([0, 2])
    assert list(sub.ids) == [0, 2]
    with pytest.raises(KeyError):
        fld.subset([1])  # id 1 was dropped as a coincident duplicate


# ---------------------------------------------------------------------------
# downselect
# ---------------------------------------------------------------------------


def test_downselect_farthest_point_returns_n_unique_sorted_indices():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-50_000.0, 50_000.0, size=(60, 2))
    fld = HeliostatField(x_mm=xy[:, 0], y_mm=xy[:, 1], ids=np.arange(60))
    idx = downselect(fld, 12, method="farthest_point")
    assert idx.size == 12
    assert len(set(idx.tolist())) == 12
    assert list(idx) == sorted(idx.tolist())


def test_downselect_uniform_returns_n_unique_indices():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-50_000.0, 50_000.0, size=(60, 2))
    fld = HeliostatField(x_mm=xy[:, 0], y_mm=xy[:, 1], ids=np.arange(60))
    idx = downselect(fld, 12, method="uniform", seed=3)
    assert idx.size == 12
    assert len(set(idx.tolist())) == 12


def test_downselect_n_ge_len_returns_everything():
    fld = HeliostatField(x_mm=np.zeros(5), y_mm=np.zeros(5), ids=np.arange(5))
    idx = downselect(fld, 10)
    np.testing.assert_array_equal(idx, np.arange(5))


def test_downselect_rejects_nonpositive_n():
    fld = HeliostatField(x_mm=np.zeros(5), y_mm=np.zeros(5), ids=np.arange(5))
    with pytest.raises(ValueError):
        downselect(fld, 0)


def test_downselect_unknown_method_raises():
    fld = HeliostatField(x_mm=np.zeros(5), y_mm=np.zeros(5), ids=np.arange(5))
    with pytest.raises(ValueError):
        downselect(fld, 2, method="nonsense")
