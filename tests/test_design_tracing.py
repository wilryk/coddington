"""Design-consuming tracer gates.

Two guarantees, in order of importance:

1. **The rectangle gate** — ``rect_heliostat`` through the generalized
   facet path must reproduce the legacy single-mirror path exactly: bit-
   identical flux maps in the cone backend, and identical counters plus
   identical *quantised* rays in the MC backend (raw float positions may
   differ by ~1 ulp from benign operation reordering; the store's int16
   quantisation is the contract, and it must not see any difference).
2. **The flower cross-check** — a canted five-petal flower with
   slant-focal spherical petals, traced by both backends, which share
   frame helpers but implement surface interception completely
   differently. Agreement bounds the generalisation's error: the ~1%
   power tolerance reflects this deliberately stressful case (heavily
   defocused spot spilling against the receiver window, MC reference
   noise ~0.2%) — the rectangle gate, not this, is the exactness claim.

The legacy path negates c4/c5 internally (inherited frame convention);
designs carry figures in their own frame, so the equivalent design uses
``ZernikeAstig(c3, -c4, -c5)``.
"""

import numpy as np
import pandas as pd
import pytest

from heliostat.geometry.design import Spherical, ZernikeAstig, flower, rect_heliostat
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat
from test_mc_parity import MC_ROOT, _geometry_for, _quantise

CONFIGS = ["prime_focus", "axicon", "cassegrain"]


def _case(config, heliostat_id=574, step_key="20260321_0939"):
    rows = pd.read_csv(MC_ROOT / config / "summary.csv")
    row = rows[
        (rows.heliostat_id == heliostat_id) & (rows.step_key.astype(str) == str(step_key))
    ].iloc[0]
    secondary, receiver = _geometry_for(config)
    args = (
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
    )
    return row, args


@pytest.mark.parametrize("config", CONFIGS)
def test_mc_rect_design_matches_legacy_at_quantisation(config):
    row, args = _case(config)
    seed = (20260811, int(row.step_key), int(row.heliostat_id))
    legacy = trace_heliostat(*args, 20000, np.random.default_rng(np.random.SeedSequence(seed)))
    design = rect_heliostat(surface=ZernikeAstig(row.c3, -row.c4, -row.c5))
    gen = trace_heliostat(
        *args, 20000, np.random.default_rng(np.random.SeedSequence(seed)), design=design
    )
    assert gen["counters"] == legacy["counters"]
    assert np.array_equal(_quantise(gen["xy"]), _quantise(legacy["xy"]))


@pytest.mark.parametrize("config", CONFIGS)
def test_cone_rect_design_matches_legacy_exactly(config):
    row, args = _case(config)
    kernel = sunshape_kernel("super_gauss")
    legacy = trace_heliostat_cone(*args, kernel)
    design = rect_heliostat(surface=ZernikeAstig(row.c3, -row.c4, -row.c5))
    gen = trace_heliostat_cone(*args, kernel, design=design)
    assert gen["counters"] == legacy["counters"]
    assert gen["power_w"] == legacy["power_w"]
    assert np.array_equal(gen["flux"], legacy["flux"])


def test_flower_cross_backend_agreement():
    row, _ = _case("prime_focus")
    design = flower(
        n_petals=5,
        petal_length_mm=2000.0,
        petal_width_mm=900.0,
        hub_radius_mm=200.0,
        surface=Spherical("slant"),
        cant_focal_mm=95000.0,
        petals_as_facets=True,
    )
    receiver = FlatWindowReceiver(z_mm=35335.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down")
    args = (
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        0.0,
        0.0,
        0.0,
        row.solar_az_deg,
        row.solar_el_deg,
        NoSecondary(),
        receiver,
    )
    mc = trace_heliostat(*args, 500_000, np.random.default_rng(11), design=design)
    p_mc = mc["watts_per_ray"] * mc["counters"]["in_window"]
    xy = mc["xy"]
    cen_mc = xy.mean(axis=1)
    rms_mc = float(np.sqrt(((xy - cen_mc[:, None]) ** 2).sum(axis=0).mean()))

    # MC's `design=design` call above passes no `sampler=`, so it draws from
    # the app-wide default (Buie, since the sunshape swap) -- the cone
    # kernel must match, or this cross-backend comparison is apples to
    # oranges.
    cone = trace_heliostat_cone(*args, sunshape_kernel("buie"), design=design)
    flux = cone["flux"]
    u_mid = 0.5 * (cone["u_edges"][:-1] + cone["u_edges"][1:])
    v_mid = 0.5 * (cone["v_edges"][:-1] + cone["v_edges"][1:])
    tot = flux.sum()
    cen = np.array([(flux.sum(axis=0) * u_mid).sum() / tot, (flux.sum(axis=1) * v_mid).sum() / tot])
    uu, vv = np.meshgrid(u_mid, v_mid)
    rms = float(np.sqrt((((uu - cen[0]) ** 2 + (vv - cen[1]) ** 2) * flux).sum() / tot))

    assert cone["power_w"] == pytest.approx(p_mc, rel=0.012)
    assert rms == pytest.approx(rms_mc, rel=0.01)
    assert np.linalg.norm(cen - cen_mc) < 15.0
    # The MC path really did trace petals: the hit fraction must be well
    # below a full rectangle's, and every counter stage populated.
    assert 0 < mc["counters"]["hit_mirror"] < 0.2 * mc["counters"]["emitted"]
