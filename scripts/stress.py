"""Combinatorial stress harness for the Coddington web API.

Drives the FastAPI app in-process (:class:`fastapi.testclient.TestClient`,
no server, no browser) with seeded, randomly-combined parameter payloads and
judges every response against the rules in ``docs/stress-test-plan.md``:
no 5xx, no NaN/Inf, plausible physics, cross-endpoint agreement, library
round-trips, and a handful of monotonic sanity checks. Every finding prints
the exact JSON payload that produced it, so it replays as a pasted request.

Usage::

    python scripts/stress.py --quick            # a few hundred cases, minutes
    python scripts/stress.py --full             # thousands of cases
    python scripts/stress.py --focus custom_polygons
    python scripts/stress.py --quick --seed 7

Heliostat counts are deliberately kept small (1-30, plus the four fixed
scale/cap cases) and Monte Carlo ray budgets are capped low -- the point is
thousands of cheap cases, not a few expensive ones. The 643-heliostat
manuscript field is never traced here.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The library/setups stores read HELIOSTAT_LIBRARY_DIR / HELIOSTAT_SETUPS_DIR
# at call time, not at import time -- setting these before the first request
# keeps a stress run's library churn out of the user's real saved designs.
_SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="heliostat-stress-"))
os.environ.setdefault("HELIOSTAT_LIBRARY_DIR", str(_SCRATCH_DIR / "library"))
os.environ.setdefault("HELIOSTAT_SETUPS_DIR", str(_SCRATCH_DIR / "setups"))

import numpy as np  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.web.app import (  # noqa: E402
    AXICON_APERTURE_RADIUS_MM,
    AXICON_APEX_HEIGHT_MM,
    AXICON_HALF_ANGLE_DEG,
    AXICON_RECEIVER_Z_MM,
    CASSEGRAIN_APERTURE_RADIUS_MM,
    CASSEGRAIN_FOCUS_HEIGHT_MM,
    CASSEGRAIN_RECEIVER_Z_MM,
    CASSEGRAIN_VERTEX_Z_MM,
    MAX_FIELD_HELIOSTATS,
    MAX_GEOMETRY_HELIOSTATS,
    PRIME_FOCUS_HEIGHT_MM,
    create_app,
)

# ---------------------------------------------------------------------------
# budgets (docs/stress-test-plan.md section 1)

BUDGET_GEOMETRY_S = 2.0
BUDGET_TRACE_S = 60.0
BUDGET_DAY_STEP_S = 30.0

RECT_DESIGN = {"type": "rect", "width_mm": 5000.0, "height_mm": 3000.0}


# ---------------------------------------------------------------------------
# findings


@dataclass
class Finding:
    severity: str  # crash | hang | wrong | other
    fclass: str  # 5xx | nan | physics | disagreement | roundtrip | monotonic
    # | validation-gap | timing
    area: str
    endpoint: str
    message: str
    payload: Any
    case_id: int
    elapsed_s: float | None = None
    budget_s: float | None = None
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """Rough de-duplication key: same class/area/endpoint/message shape."""
        return (self.severity, self.fclass, self.area, self.endpoint, self.message[:80])


SEVERITY_ORDER = {"crash": 0, "hang": 1, "wrong": 2, "other": 3}


class Findings:
    def __init__(self) -> None:
        self.items: list[Finding] = []
        self._seen: set[tuple] = set()
        self._counter = itertools.count(1)
        self.cases_run = 0
        self.calls_made = 0

    def next_case_id(self) -> int:
        return next(self._counter)

    def add(self, finding: Finding, *, cap_per_key: int = 5) -> None:
        key = finding.key()
        n = sum(1 for f in self.items if f.key() == key)
        if n >= cap_per_key:
            return
        self.items.append(finding)

    def by_severity(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.items:
            out.setdefault(f.severity, []).append(f)
        return out


# ---------------------------------------------------------------------------
# response inspection


def _walk_numbers(obj, path=""):
    """Yield (path, value) for every int/float leaf in a decoded JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield path, obj


def find_non_finite(obj) -> list[str]:
    bad = []
    for path, value in _walk_numbers(obj):
        try:
            if not math.isfinite(value):
                bad.append(f"{path}={value!r}")
        except (TypeError, ValueError):
            continue
    return bad


def dig(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# HTTP call wrapper


class Caller:
    """Thin wrapper: times a call, never lets a 5xx raise into the harness."""

    def __init__(self, client: TestClient):
        self.client = client

    def call(self, method: str, path: str, payload=None, **kw):
        t0 = time.perf_counter()
        try:
            if method == "GET":
                resp = self.client.get(path, **kw)
            elif method == "DELETE":
                resp = self.client.delete(path, **kw)
            elif method == "POST":
                resp = self.client.post(path, json=payload, **kw)
            else:  # pragma: no cover
                raise ValueError(method)
        except Exception as exc:  # noqa: BLE001 -- a transport-level crash is still a finding
            elapsed = time.perf_counter() - t0
            return _FakeResponse(599, {"detail": f"{type(exc).__name__}: {exc}"}), elapsed
        elapsed = time.perf_counter() - t0
        return resp, elapsed


class _FakeResponse:
    """Stand-in for a response when the call itself raised (should not
    happen with raise_server_exceptions=False, but belt and braces)."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


# ---------------------------------------------------------------------------
# generic judging


def judge_common(
    findings: Findings,
    *,
    case_id: int,
    area: str,
    endpoint: str,
    payload: Any,
    resp,
    elapsed_s: float,
    budget_s: float | None,
) -> dict | None:
    """Crash / NaN / timing checks common to every endpoint.

    Returns the decoded JSON body on success (2xx with a parseable body),
    else ``None`` -- callers skip further physics checks when this is None.
    """
    findings.calls_made += 1
    status = resp.status_code

    if status >= 500:
        findings.add(
            Finding(
                "crash",
                "5xx",
                area,
                endpoint,
                f"{status}: {_short(resp)}",
                payload,
                case_id,
                elapsed_s,
                budget_s,
            )
        )
        return None

    content_type = resp.headers.get("content-type", "")
    if status < 400 and "json" not in content_type:
        # Endpoints that legitimately return something other than JSON on
        # success (design/preview and design/sag return image/png, the CSV
        # exports return text/csv) -- not a parse failure, just not JSON.
        return None

    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        if status < 400:
            findings.add(
                Finding(
                    "crash",
                    "unparseable-response",
                    area,
                    endpoint,
                    f"{status} (content-type={content_type!r}): body did not parse as JSON ({exc})",
                    payload,
                    case_id,
                    elapsed_s,
                    budget_s,
                )
            )
        return None

    if status >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        readable = (isinstance(detail, str) and detail.strip()) or (
            isinstance(detail, list) and len(detail) > 0
        )
        if not readable:
            findings.add(
                Finding(
                    "wrong",
                    "unreadable-422",
                    area,
                    endpoint,
                    f"{status} with no readable 'detail' message: {body!r}",
                    payload,
                    case_id,
                    elapsed_s,
                    budget_s,
                )
            )
        return None

    # 2xx: NaN/Inf anywhere, and the timing budget.
    bad = find_non_finite(body)
    if bad:
        findings.add(
            Finding(
                "wrong",
                "nan",
                area,
                endpoint,
                "non-finite value(s) in response: " + ", ".join(bad[:6]),
                payload,
                case_id,
                elapsed_s,
                budget_s,
            )
        )

    if budget_s is not None and elapsed_s > budget_s:
        findings.add(
            Finding(
                "hang",
                "timing",
                area,
                endpoint,
                f"took {elapsed_s:.2f}s, budget is {budget_s:.1f}s",
                payload,
                case_id,
                elapsed_s,
                budget_s,
            )
        )

    return body


def _short(resp) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])[:400]
        return str(body)[:400]
    except Exception:  # noqa: BLE001
        return (resp.text or "")[:400]


def add_wrong(findings, case_id, area, endpoint, fclass, message, payload, elapsed_s=None, budget_s=None, **extra):
    findings.add(
        Finding("wrong", fclass, area, endpoint, message, payload, case_id, elapsed_s, budget_s, extra)
    )


# ---------------------------------------------------------------------------
# physics judges


#: The cone backends (ultra_fast/fast_accurate) integrate incident and
#: collected power through two different quadratures (mirror-side sampling
#: vs. receiver-side accumulation), so a tiny power_w > incident_power_w
#: excess is expected discretisation noise, not a bug -- modes.py documents
#: total power as accurate to about +-0.1% for these backends. Flagging
#: anything past a few times that margin still catches a real violation
#: (a 17x excess has been observed) without flooding the report with noise
#: from every plain trace.
CONE_POWER_REL_TOL = 2.0e-3
MC_POWER_REL_TOL = 1.0e-6


def judge_trace_body(findings, case_id, area, endpoint, payload, body, elapsed_s, budget_s):
    power = body.get("power_w")
    incident = body.get("incident_power_w")
    peak = body.get("peak_flux_kw_m2")
    rms = body.get("rms_radius_mm")
    mode = payload.get("mode")
    rel_tol = MC_POWER_REL_TOL if mode == "monte_carlo" else CONE_POWER_REL_TOL

    if power is not None and power < -1e-6:
        add_wrong(findings, case_id, area, endpoint, "physics", f"power_w={power} < 0", payload, elapsed_s, budget_s)
    if incident is not None:
        if incident < -1e-6:
            add_wrong(
                findings, case_id, area, endpoint, "physics",
                f"incident_power_w={incident} < 0", payload, elapsed_s, budget_s,
            )
        if power is not None and incident is not None and power > incident * (1.0 + rel_tol) + 1e-9:
            excess = (power / incident - 1.0) if incident > 0 else float("inf")
            add_wrong(
                findings, case_id, area, endpoint, "physics",
                f"power_w={power} > incident_power_w={incident} "
                f"({excess * 100:.3f}% over, tolerance {rel_tol * 100:.2f}%)",
                payload, elapsed_s, budget_s,
            )
    if peak is not None and peak < -1e-9:
        add_wrong(findings, case_id, area, endpoint, "physics", f"peak_flux_kw_m2={peak} < 0", payload, elapsed_s, budget_s)
    # rms radius > 0 whenever real power lands, for the deterministic cone
    # backends only -- MC with a handful of rays can coincidentally put every
    # hit at one point, which is not a bug.
    if (
        mode in ("ultra_fast", "fast_accurate")
        and power is not None
        and power > 1e-6
        and rms is not None
        and rms <= 0.0
    ):
        add_wrong(
            findings, case_id, area, endpoint, "physics",
            f"power_w={power} > 0 but rms_radius_mm={rms}", payload, elapsed_s, budget_s,
        )


def judge_field_body(findings, case_id, area, endpoint, payload, body, elapsed_s, budget_s):
    judge_trace_body(findings, case_id, area, endpoint, payload, body, elapsed_s, budget_s)
    rows = body.get("heliostats") or []
    n = body.get("n_heliostats")
    if n is not None and n != len(rows):
        add_wrong(
            findings, case_id, area, endpoint, "disagreement",
            f"n_heliostats={n} but heliostats has {len(rows)} rows", payload, elapsed_s, budget_s,
        )
    for r in rows:
        for k in ("eta_shade", "eta_block", "eta"):
            v = r.get(k)
            if v is not None and not (-1e-9 <= v <= 1.0 + 1e-6):
                add_wrong(
                    findings, case_id, area, endpoint, "physics",
                    f"heliostat id={r.get('id')} {k}={v} outside [0,1]", payload, elapsed_s, budget_s,
                )
    for k in ("eta_min", "eta_median", "eta_max"):
        v = body.get(k)
        if v is not None and not (-1e-9 <= v <= 1.0 + 1e-6):
            add_wrong(findings, case_id, area, endpoint, "physics", f"{k}={v} outside [0,1]", payload, elapsed_s, budget_s)
    if body.get("eta_min") is not None and body.get("eta_max") is not None:
        if body["eta_min"] > body["eta_max"] + 1e-9:
            add_wrong(
                findings, case_id, area, endpoint, "physics",
                f"eta_min={body['eta_min']} > eta_max={body['eta_max']}", payload, elapsed_s, budget_s,
            )
    power = body.get("power_w")
    if power is not None and rows:
        row_sum = sum(r.get("power_w") or 0.0 for r in rows)
        if abs(power - row_sum) > 1e-6 * max(1.0, abs(power)) + 1e-6:
            add_wrong(
                findings, case_id, area, endpoint, "disagreement",
                f"power_w={power} != sum(heliostat power_w)={row_sum}", payload, elapsed_s, budget_s,
            )


# ---------------------------------------------------------------------------
# rng helpers


def pick(rng: random.Random, seq):
    return rng.choice(seq)


def maybe(rng: random.Random, p, value, otherwise=None):
    return value if rng.random() < p else otherwise


SURFACES = ["twisting", "spherical", "flat"]


def mk_rect_design(rng: random.Random) -> dict:
    return {
        "type": "rect",
        "width_mm": pick(rng, [10.0, 500.0, 3000.0, 5000.0, 12000.0, 1.0e6]),
        "height_mm": pick(rng, [10.0, 500.0, 2000.0, 3000.0, 8000.0, 1.0e6]),
        "surface": pick(rng, SURFACES),
        "slope_error_mrad": pick(rng, [0.0, 0.0, 5.0, 50.0]),
        "specularity_mrad": pick(rng, [0.0, 0.0, 3.0, 30.0]),
        "reflectance": pick(rng, [1.0, 1.0, 0.9, 0.5, 1.0e-3]),
    }


def mk_grid_design(rng: random.Random) -> dict:
    return {
        "type": "grid",
        "n_u": pick(rng, [1, 2, 3, 5]),
        "n_v": pick(rng, [1, 2, 3, 5]),
        "facet_w_mm": pick(rng, [200.0, 1200.0, 3000.0]),
        "facet_h_mm": pick(rng, [200.0, 1000.0, 2500.0]),
        "gap_mm": pick(rng, [0.0, 10.0, 100.0]),
        "cant_focal_mm": pick(rng, [None, 0.0, 500.0, 1.0e7]),
        "facet_focal_mm": pick(rng, [None, 0.0, 300.0, 1.0e7]),
        "surface": pick(rng, SURFACES),
        "slope_error_mrad": pick(rng, [0.0, 5.0]),
        "specularity_mrad": pick(rng, [0.0, 3.0]),
        "reflectance": pick(rng, [1.0, 0.9]),
    }


def mk_flower_design(rng: random.Random) -> dict:
    length = pick(rng, [800.0, 2000.0, 4000.0])
    width = min(pick(rng, [400.0, 900.0, 1500.0]), 1.9 * length)
    return {
        "type": "flower",
        "n_petals": pick(rng, [3, 5, 8]),
        "petal_length_mm": length,
        "petal_width_mm": width,
        "hub_radius_mm": pick(rng, [0.0, 200.0, 1000.0]),
        "cant_focal_mm": pick(rng, [None, 0.0, 500.0, 1.0e7]),
        "facet_focal_mm": pick(rng, [None, 0.0, 300.0]),
        "surface": pick(rng, SURFACES),
        "slope_error_mrad": pick(rng, [0.0, 5.0]),
        "specularity_mrad": pick(rng, [0.0, 3.0]),
        "reflectance": pick(rng, [1.0, 0.9]),
    }


def mk_any_design(rng: random.Random) -> dict:
    kind = pick(rng, ["rect", "grid", "flower"])
    if kind == "rect":
        return mk_rect_design(rng)
    if kind == "grid":
        return mk_grid_design(rng)
    return mk_flower_design(rng)


# -- custom polygons ---------------------------------------------------------


def poly_triangle(_rng) -> list:
    return [(0.0, 0.0), (2000.0, 0.0), (1000.0, 1800.0)]


def poly_many(_rng, n: int = 60) -> list:
    r = 2500.0
    return [
        (round(r * math.cos(2 * math.pi * i / n), 3), round(r * math.sin(2 * math.pi * i / n), 3))
        for i in range(n)
    ]


def poly_self_intersecting(_rng) -> list:
    # A bowtie: edges 0-1 and 2-3 cross.
    return [(-2000.0, -1000.0), (2000.0, 1000.0), (2000.0, -1000.0), (-2000.0, 1000.0)]


def poly_sliver(_rng) -> list:
    return [(0.0, 0.0), (10000.0, 0.0), (10000.0, 0.5), (0.0, 0.5)]


def poly_tiny(_rng) -> list:
    return [(0.0, 0.0), (1.0, 0.0), (0.5, 0.9)]


def poly_huge(_rng) -> list:
    r = 5_000_000.0
    return [(-r, -r), (r, -r), (r, r), (-r, r)]


def poly_symmetric_hex(_rng) -> list:
    r = 2500.0
    return [
        (round(r * math.cos(math.radians(a)), 3), round(r * math.sin(math.radians(a)), 3))
        for a in range(0, 360, 60)
    ]


def poly_asymmetric_l(_rng) -> list:
    return [(0.0, 0.0), (3000.0, 0.0), (3000.0, 800.0), (800.0, 800.0), (800.0, 2500.0), (0.0, 2500.0)]


CUSTOM_POLY_BUILDERS = [
    poly_triangle,
    poly_many,
    poly_self_intersecting,
    poly_sliver,
    poly_tiny,
    poly_huge,
    poly_symmetric_hex,
    poly_asymmetric_l,
]


def mk_custom_design(rng: random.Random) -> dict:
    builder = pick(rng, CUSTOM_POLY_BUILDERS)
    return {
        "type": "custom",
        "vertices_mm": builder(rng),
        "surface": pick(rng, SURFACES),
        "slope_error_mrad": pick(rng, [0.0, 5.0]),
        "specularity_mrad": pick(rng, [0.0, 3.0]),
        "reflectance": pick(rng, [1.0, 0.9]),
    }


# -- optics params ------------------------------------------------------------


def mk_prime_focus_params(rng: random.Random) -> dict:
    variant = pick(rng, ["default", "cylinder", "frustum", "frustum_collapsed", "offset", "offset_recv", "tiny_focus"])
    if variant == "default":
        return {}
    if variant == "cylinder":
        return {
            "receiver_type": "cylinder",
            "cylinder_radius_mm": pick(rng, [0.5, 3000.0, 1.0e6]),
            "cylinder_height_mm": pick(rng, [0.5, 6000.0, 1.0e6]),
        }
    if variant == "frustum":
        return {
            "receiver_type": "frustum",
            "frustum_top_radius_mm": pick(rng, [500.0, 2500.0, 1.0e5]),
            "frustum_bottom_radius_mm": pick(rng, [500.0, 4000.0, 1.0e5]),
            "frustum_height_mm": pick(rng, [0.5, 6000.0, 1.0e5]),
        }
    if variant == "frustum_collapsed":
        r = pick(rng, [1500.0, 3000.0])
        return {"receiver_type": "frustum", "frustum_top_radius_mm": r, "frustum_bottom_radius_mm": r}
    if variant == "offset":
        return {"receiver_center_x_mm": pick(rng, [-1.0e5, 1.0e5]), "receiver_center_y_mm": pick(rng, [-1.0e5, 1.0e5])}
    if variant == "offset_recv":
        return {"aperture_to_receiver_mm": pick(rng, [0.0, 500.0, 1.0e5])}
    return {"focus_height_mm": pick(rng, [1.0, 100.0, PRIME_FOCUS_HEIGHT_MM])}


def mk_axicon_params(rng: random.Random) -> dict:
    apex = pick(rng, [10.0, AXICON_APEX_HEIGHT_MM, 1.0e6])
    return {
        "apex_height_mm": apex,
        "half_angle_deg": pick(rng, [0.001, 1.0, AXICON_HALF_ANGLE_DEG, 89.999]),
        "aperture_radius_mm": pick(rng, [0.001, AXICON_APERTURE_RADIUS_MM, 1.0e7]),
        "receiver_z_mm": pick(rng, [0.001, AXICON_RECEIVER_Z_MM, apex * 0.999, apex * 1.5]),
    }


def mk_cassegrain_params(rng: random.Random) -> dict:
    variant = pick(rng, ["default", "boundary_near_flat", "aperture_extreme"])
    if variant == "boundary_near_flat":
        vertex = CASSEGRAIN_VERTEX_Z_MM
        focus = CASSEGRAIN_FOCUS_HEIGHT_MM
        d1 = focus - vertex
        # Push the receiver just past the "solves flat" boundary
        # (|d2| == d1) so the relay is barely a valid hyperboloid.
        receiver = vertex - d1 * pick(rng, [1.001, 1.02, 1.3])
        return {"vertex_z_mm": vertex, "focus_height_mm": focus, "receiver_z_mm": receiver}
    if variant == "aperture_extreme":
        return {"aperture_radius_mm": pick(rng, [0.5, 1.0e7])}
    return {}


def mk_optics(rng: random.Random) -> tuple[str, dict]:
    optics = pick(rng, ["prime_focus", "axicon", "cassegrain"])
    params = {
        "prime_focus": mk_prime_focus_params,
        "axicon": mk_axicon_params,
        "cassegrain": mk_cassegrain_params,
    }[optics](rng)
    return optics, params


SUN_ELEVATIONS = [-10.0, -1.9, -0.01, 0.0, 0.01, 1.9, 5.0, 20.0, 45.0, 89.9, 90.0]
SUN_AZIMUTHS = [0.0, 0.001, 90.0, 180.0, 270.0, 359.999, 360.0]

FAR_POSITIONS = [
    (0.0, 0.0),
    (1.0e-6, 1.0e-6),
    (1.0, 1.0),
    (0.0, -50000.0),
    (0.0, 89609.0),
    (1.0e8, -1.0e8),
    (-1.0e8, 0.0),
    (-30000.0, -30000.0),
]

MODES = ["ultra_fast", "fast_accurate", "monte_carlo"]


def mk_trace_payload(rng: random.Random, *, design=None, optics=None, params=None) -> dict:
    design = design if design is not None else mk_any_design(rng)
    if optics is None:
        optics, params = mk_optics(rng)
    mode = pick(rng, MODES)
    payload = {
        "design": design,
        "mode": mode,
        "optics": optics,
        "solar_az_deg": pick(rng, SUN_AZIMUTHS),
        "solar_el_deg": pick(rng, SUN_ELEVATIONS),
        "optics_params": params,
    }
    x, y = pick(rng, FAR_POSITIONS)
    payload["heliostat_x_mm"] = x
    payload["heliostat_y_mm"] = y
    if mode == "monte_carlo":
        payload["n_rays"] = pick(rng, [100, 200, 1000, 5000])
    return payload


# ---------------------------------------------------------------------------
# area runners


def run_design_matrix(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        payload = mk_trace_payload(rng)
        resp, elapsed = caller.call("POST", "/api/trace", payload)
        body = judge_common(
            findings, case_id=cid, area="design_matrix", endpoint="POST /api/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if body is not None:
            judge_trace_body(findings, cid, "design_matrix", "POST /api/trace", payload, body, elapsed, BUDGET_TRACE_S)


def run_custom_polygons(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        design = mk_custom_design(rng)
        optics, params = mk_optics(rng)
        payload = mk_trace_payload(rng, design=design, optics=optics, params=params)
        resp, elapsed = caller.call("POST", "/api/trace", payload)
        body = judge_common(
            findings, case_id=cid, area="custom_polygons", endpoint="POST /api/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if body is not None:
            judge_trace_body(findings, cid, "custom_polygons", "POST /api/trace", payload, body, elapsed, BUDGET_TRACE_S)

        # Preview and scene/geometry take the same design through different
        # code paths -- cheap to hit both with the same shape.
        cid2 = findings.next_case_id()
        resp2, elapsed2 = caller.call("POST", "/api/design/preview", {"design": design})
        judge_common(
            findings, case_id=cid2, area="custom_polygons", endpoint="POST /api/design/preview",
            payload={"design": design}, resp=resp2, elapsed_s=elapsed2, budget_s=BUDGET_GEOMETRY_S,
        )


def run_degenerate_optics(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        optics, params = mk_optics(rng)
        design = mk_rect_design(rng)
        payload = mk_trace_payload(rng, design=design, optics=optics, params=params)
        resp, elapsed = caller.call("POST", "/api/trace", payload)
        body = judge_common(
            findings, case_id=cid, area="degenerate_optics", endpoint="POST /api/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if body is not None:
            judge_trace_body(findings, cid, "degenerate_optics", "POST /api/trace", payload, body, elapsed, BUDGET_TRACE_S)

    # A few explicit, deliberately-invalid geometries that MUST 422 cleanly,
    # never 500: receiver above the apex, a non-hyperboloid Cassegrain relay.
    explicit = [
        ("axicon", {"apex_height_mm": 20000.0, "receiver_z_mm": 25000.0}),  # above apex
        ("axicon", {"apex_height_mm": 20000.0, "receiver_z_mm": 20000.0}),  # equal to apex
        ("cassegrain", {"vertex_z_mm": 10000.0, "focus_height_mm": 5000.0, "receiver_z_mm": -1.0}),  # focus below vertex
        (
            "cassegrain",
            {"vertex_z_mm": CASSEGRAIN_VERTEX_Z_MM, "focus_height_mm": CASSEGRAIN_FOCUS_HEIGHT_MM, "receiver_z_mm": CASSEGRAIN_VERTEX_Z_MM - (CASSEGRAIN_FOCUS_HEIGHT_MM - CASSEGRAIN_VERTEX_Z_MM)},
        ),  # exactly the |d2|==d1 boundary -- not a hyperboloid
        ("prime_focus", {"focus_height_mm": 1.0, "receiver_type": "cylinder", "cylinder_height_mm": 10.0}),  # z_bot<=0
    ]
    for optics, params in explicit:
        cid = findings.next_case_id()
        payload = mk_trace_payload(rng, design=RECT_DESIGN, optics=optics, params=params)
        payload["solar_el_deg"] = 45.0
        resp, elapsed = caller.call("POST", "/api/trace", payload)
        judge_common(
            findings, case_id=cid, area="degenerate_optics", endpoint="POST /api/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if resp.status_code < 400:
            add_wrong(
                findings, cid, "degenerate_optics", "POST /api/trace", "validation-gap",
                f"expected a clean 422 for an impossible geometry, got {resp.status_code}", payload,
            )


def run_sun_positions(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        payload = mk_trace_payload(rng, design=mk_rect_design(rng))
        el = payload["solar_el_deg"]
        resp, elapsed = caller.call("POST", "/api/trace", payload)
        body = judge_common(
            findings, case_id=cid, area="sun_positions", endpoint="POST /api/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if el <= 0.0 and resp.status_code != 422:
            add_wrong(
                findings, cid, "sun_positions", "POST /api/trace", "validation-gap",
                f"solar_el_deg={el} <= 0 should be a clean 422, got {resp.status_code}", payload,
            )
        if body is not None:
            judge_trace_body(findings, cid, "sun_positions", "POST /api/trace", payload, body, elapsed, BUDGET_TRACE_S)

        # /api/scene/geometry must accept the sun below the horizon instead
        # of erroring, and say so.
        cid2 = findings.next_case_id()
        geo_payload = {
            "design": payload["design"],
            "optics": payload["optics"],
            "optics_params": payload["optics_params"],
            "solar_az_deg": payload["solar_az_deg"],
            "solar_el_deg": el,
            "heliostat_x_mm": payload["heliostat_x_mm"],
            "heliostat_y_mm": payload["heliostat_y_mm"],
        }
        resp2, elapsed2 = caller.call("POST", "/api/scene/geometry", geo_payload)
        body2 = judge_common(
            findings, case_id=cid2, area="sun_positions", endpoint="POST /api/scene/geometry",
            payload=geo_payload, resp=resp2, elapsed_s=elapsed2, budget_s=BUDGET_GEOMETRY_S,
        )
        if body2 is not None and el <= 0.0 and not body2.get("sun_below_horizon"):
            add_wrong(
                findings, cid2, "sun_positions", "POST /api/scene/geometry", "wrong",
                f"solar_el_deg={el} <= 0 but sun_below_horizon is not true", geo_payload,
            )


def mk_layout(rng: random.Random, max_n: int = 30, *, cap: int = MAX_FIELD_HELIOSTATS) -> dict:
    kind = pick(rng, ["fermat", "positions", "radial_stagger"])
    if kind == "fermat":
        n = pick(rng, [1, 2, 5, 10, min(30, max_n)])
        layout = {"type": "fermat", "n": n}
        if rng.random() < 0.4:
            layout["a_m"] = pick(rng, [0.1, 4.5, 50.0])
        if rng.random() < 0.3:
            layout["r_min_m"] = pick(rng, [0.0, 5.0])
            layout["r_max_m"] = pick(rng, [10.0, 200.0])
        return layout
    if kind == "positions":
        n = pick(rng, [1, 2, 3, 8, min(20, max_n)])
        pts = []
        for i in range(n):
            x, y = pick(rng, FAR_POSITIONS)
            pts.append([x, y])
        if rng.random() < 0.25 and n >= 2:
            pts[-1] = list(pts[0])  # coincident heliostats
        return {"type": "positions", "xy_mm": pts}
    # radial_stagger: small, deliberately varied bands.
    band_count = pick(rng, [4, 6, 8])
    n_rings = pick(rng, [1, 2])
    ring_radii = [pick(rng, [30.0, 40.0, 60.0]) for _ in range(n_rings)]
    return {
        "type": "radial_stagger",
        "band_counts": [band_count],
        "band_ring_counts": [n_rings],
        "ring_radii_m": ring_radii,
    }


def run_field_positions(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        design = mk_any_design(rng)
        optics, params = mk_optics(rng)
        layout = mk_layout(rng, max_n=20)
        payload = {
            "design": design,
            "mode": pick(rng, MODES),
            "optics": optics,
            "solar_az_deg": pick(rng, SUN_AZIMUTHS),
            "solar_el_deg": pick(rng, [1.9, 5.0, 20.0, 45.0, 89.9]),
            "optics_params": params,
            "layout": layout,
        }
        if payload["mode"] == "monte_carlo":
            payload["n_rays"] = pick(rng, [100, 200, 1000])
        resp, elapsed = caller.call("POST", "/api/field/trace", payload)
        body = judge_common(
            findings, case_id=cid, area="field_positions", endpoint="POST /api/field/trace",
            payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if body is not None:
            judge_field_body(findings, cid, "field_positions", "POST /api/field/trace", payload, body, elapsed, BUDGET_TRACE_S)


def run_field_trace_general(caller, findings, rng, n):
    """Field trace x /api/scene/geometry cross-checks for the same layout."""
    for _ in range(n):
        cid = findings.next_case_id()
        design = mk_any_design(rng)
        optics, params = mk_optics(rng)
        layout = mk_layout(rng, max_n=25)
        el = pick(rng, [1.9, 5.0, 20.0, 45.0, 89.9])
        az = pick(rng, SUN_AZIMUTHS)
        trace_payload = {
            "design": design,
            "mode": pick(rng, ["ultra_fast", "fast_accurate"]),
            "optics": optics,
            "solar_az_deg": az,
            "solar_el_deg": el,
            "optics_params": params,
            "layout": layout,
        }
        resp, elapsed = caller.call("POST", "/api/field/trace", trace_payload)
        body = judge_common(
            findings, case_id=cid, area="field_trace", endpoint="POST /api/field/trace",
            payload=trace_payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S,
        )
        if body is not None:
            judge_field_body(findings, cid, "field_trace", "POST /api/field/trace", trace_payload, body, elapsed, BUDGET_TRACE_S)

        cid2 = findings.next_case_id()
        geo_payload = {
            "design": design,
            "optics": optics,
            "optics_params": params,
            "solar_az_deg": az,
            "solar_el_deg": el,
            "layout": layout,
        }
        resp2, elapsed2 = caller.call("POST", "/api/scene/geometry", geo_payload)
        body2 = judge_common(
            findings, case_id=cid2, area="field_trace", endpoint="POST /api/scene/geometry",
            payload=geo_payload, resp=resp2, elapsed_s=elapsed2, budget_s=BUDGET_GEOMETRY_S,
        )
        if body is not None and body2 is not None:
            n1 = body.get("n_heliostats")
            n2 = len(body2.get("heliostats") or [])
            if n1 is not None and n1 != n2:
                add_wrong(
                    findings, cid2, "field_trace", "POST /api/scene/geometry", "disagreement",
                    f"field/trace n_heliostats={n1} but scene/geometry reports {n2}", geo_payload,
                )
            r1 = dig(body, "scene.receiver")
            r2 = dig(body2, "receiver")
            if r1 is not None and r2 is not None and r1 != r2:
                add_wrong(
                    findings, cid2, "field_trace", "POST /api/scene/geometry", "disagreement",
                    "field/trace scene.receiver != scene/geometry receiver for identical optics_params",
                    geo_payload, extra={"trace_receiver": r1, "geometry_receiver": r2},
                )
            o1 = body.get("optics_resolved")
            o2 = body2.get("optics_resolved")
            if o1 is not None and o2 is not None and o1 != o2:
                add_wrong(
                    findings, cid2, "field_trace", "POST /api/scene/geometry", "disagreement",
                    "optics_resolved differs between /api/field/trace and /api/scene/geometry", geo_payload,
                )


def run_cross_agreement(caller, findings, rng, n):
    """Single trace vs. a one-heliostat field at the same position -- must
    be the same physics through two entry points (docs/stress-test-plan.md's
    'Disagreement' row, and the existing test_field_of_one_equals_the_single_trace
    invariant, exercised over many more combinations)."""
    for _ in range(n):
        cid = findings.next_case_id()
        design = mk_any_design(rng)
        optics, params = mk_optics(rng)
        x, y = pick(rng, [p for p in FAR_POSITIONS if p != (0.0, 0.0)])
        el = pick(rng, [5.0, 20.0, 45.0, 80.0])
        az = pick(rng, SUN_AZIMUTHS)
        mode = pick(rng, ["ultra_fast", "fast_accurate"])
        single = {
            "design": design, "mode": mode, "optics": optics,
            "solar_az_deg": az, "solar_el_deg": el, "optics_params": params,
            "heliostat_x_mm": x, "heliostat_y_mm": y,
        }
        field_payload = {
            "design": design, "mode": mode, "optics": optics,
            "solar_az_deg": az, "solar_el_deg": el, "optics_params": params,
            "layout": {"type": "positions", "xy_mm": [[x, y]]},
        }
        r1, e1 = caller.call("POST", "/api/trace", single)
        b1 = judge_common(findings, case_id=cid, area="cross_agreement", endpoint="POST /api/trace",
                           payload=single, resp=r1, elapsed_s=e1, budget_s=BUDGET_TRACE_S)
        cid2 = findings.next_case_id()
        r2, e2 = caller.call("POST", "/api/field/trace", field_payload)
        b2 = judge_common(findings, case_id=cid2, area="cross_agreement", endpoint="POST /api/field/trace",
                           payload=field_payload, resp=r2, elapsed_s=e2, budget_s=BUDGET_TRACE_S)
        if b1 is None or b2 is None:
            continue
        for key, tol in (("power_w", 1e-6), ("rms_radius_mm", 1e-6), ("incident_power_w", 1e-6)):
            v1, v2 = b1.get(key), b2.get(key)
            if v1 is None or v2 is None:
                if v1 != v2:
                    add_wrong(
                        findings, cid2, "cross_agreement", "POST /api/field/trace", "disagreement",
                        f"{key}: single={v1!r} field-of-one={v2!r}", field_payload,
                    )
                continue
            if abs(v1 - v2) > tol * max(1.0, abs(v1)):
                add_wrong(
                    findings, cid2, "cross_agreement", "POST /api/field/trace", "disagreement",
                    f"{key}: single={v1} field-of-one={v2} (a lone heliostat should match its own single trace)",
                    field_payload,
                )
        rs = dig(b1, "scene.receiver")
        rf = dig(b2, "scene.receiver")
        if rs is not None and rf is not None and rs != rf:
            add_wrong(
                findings, cid2, "cross_agreement", "POST /api/field/trace", "disagreement",
                "scene.receiver differs between /api/trace and a one-heliostat /api/field/trace", field_payload,
            )


# -- caps ---------------------------------------------------------------------


def run_caps(caller, findings, rng):
    # trace cap: 1000 must be accepted (timed, generous budget); 1001 must
    # refuse cleanly and cheaply.
    cid = findings.next_case_id()
    payload = {
        "design": RECT_DESIGN, "mode": "ultra_fast", "optics": "prime_focus",
        "solar_az_deg": 180.0, "solar_el_deg": 45.0,
        "layout": {"type": "fermat", "n": MAX_FIELD_HELIOSTATS},
    }
    resp, elapsed = caller.call("POST", "/api/field/trace", payload)
    judge_common(
        findings, case_id=cid, area="caps", endpoint="POST /api/field/trace",
        payload=payload, resp=resp, elapsed_s=elapsed, budget_s=None,
    )
    if resp.status_code >= 400:
        add_wrong(
            findings, cid, "caps", "POST /api/field/trace", "validation-gap",
            f"{MAX_FIELD_HELIOSTATS} heliostats (the documented cap) was refused: {resp.status_code} {_short(resp)}",
            payload, elapsed,
        )
    elif elapsed > BUDGET_TRACE_S:
        add_wrong(
            findings, cid, "caps", "POST /api/field/trace", "timing",
            f"{MAX_FIELD_HELIOSTATS}-heliostat ultra_fast field trace took {elapsed:.1f}s "
            f"(single-trace budget is {BUDGET_TRACE_S:.0f}s -- a performance cliff at the documented cap)",
            payload, elapsed,
        )

    cid = findings.next_case_id()
    over_payload = dict(payload, layout={"type": "fermat", "n": MAX_FIELD_HELIOSTATS + 1})
    resp, elapsed = caller.call("POST", "/api/field/trace", over_payload)
    judge_common(
        findings, case_id=cid, area="caps", endpoint="POST /api/field/trace",
        payload=over_payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_GEOMETRY_S,
    )
    if resp.status_code != 422:
        add_wrong(
            findings, cid, "caps", "POST /api/field/trace", "validation-gap",
            f"{MAX_FIELD_HELIOSTATS + 1} heliostats (one over the trace cap) should refuse with 422, "
            f"got {resp.status_code}", over_payload, elapsed,
        )

    # geometry cap: 10000 should be cheap; 10001 must refuse cleanly.
    cid = findings.next_case_id()
    geo_payload = {
        "design": RECT_DESIGN, "optics": "prime_focus",
        "solar_az_deg": 180.0, "solar_el_deg": 45.0,
        "layout": {"type": "fermat", "n": MAX_GEOMETRY_HELIOSTATS},
    }
    resp, elapsed = caller.call("POST", "/api/scene/geometry", geo_payload)
    judge_common(
        findings, case_id=cid, area="caps", endpoint="POST /api/scene/geometry",
        payload=geo_payload, resp=resp, elapsed_s=elapsed, budget_s=None,
    )
    if resp.status_code >= 400:
        add_wrong(
            findings, cid, "caps", "POST /api/scene/geometry", "validation-gap",
            f"{MAX_GEOMETRY_HELIOSTATS} heliostats (the documented geometry cap) was refused: "
            f"{resp.status_code} {_short(resp)}", geo_payload, elapsed,
        )
    elif elapsed > BUDGET_GEOMETRY_S * 10:
        add_wrong(
            findings, cid, "caps", "POST /api/scene/geometry", "timing",
            f"{MAX_GEOMETRY_HELIOSTATS}-heliostat geometry solve took {elapsed:.2f}s", geo_payload, elapsed,
        )

    cid = findings.next_case_id()
    over_geo = dict(geo_payload, layout={"type": "fermat", "n": MAX_GEOMETRY_HELIOSTATS + 1})
    resp, elapsed = caller.call("POST", "/api/scene/geometry", over_geo)
    judge_common(
        findings, case_id=cid, area="caps", endpoint="POST /api/scene/geometry",
        payload=over_geo, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_GEOMETRY_S,
    )
    if resp.status_code != 422:
        add_wrong(
            findings, cid, "caps", "POST /api/scene/geometry", "validation-gap",
            f"{MAX_GEOMETRY_HELIOSTATS + 1} heliostats (one over the geometry cap) should refuse with 422, "
            f"got {resp.status_code}", over_geo, elapsed,
        )


# -- monotonic sanity ----------------------------------------------------------


def run_monotonic(caller, findings, rng, n):
    for _ in range(n):
        kind = pick(rng, ["reflectance", "aperture", "field_count"])
        if kind == "reflectance":
            _check_reflectance(caller, findings, rng)
        elif kind == "aperture":
            _check_aperture(caller, findings, rng)
        else:
            _check_field_count(caller, findings, rng)


def _check_reflectance(caller, findings, rng):
    design = mk_rect_design(rng)
    design["surface"] = "twisting"
    base = {
        "design": design, "mode": "ultra_fast", "optics": "prime_focus",
        "solar_az_deg": 180.0, "solar_el_deg": pick(rng, [20.0, 45.0, 70.0]),
    }
    hi = copy.deepcopy(base)
    hi["design"] = dict(hi["design"], reflectance=1.0)
    lo = copy.deepcopy(base)
    lo["design"] = dict(lo["design"], reflectance=0.9)

    cid = findings.next_case_id()
    r1, e1 = caller.call("POST", "/api/trace", hi)
    b1 = judge_common(findings, case_id=cid, area="monotonic", endpoint="POST /api/trace",
                       payload=hi, resp=r1, elapsed_s=e1, budget_s=BUDGET_TRACE_S)
    cid2 = findings.next_case_id()
    r2, e2 = caller.call("POST", "/api/trace", lo)
    b2 = judge_common(findings, case_id=cid2, area="monotonic", endpoint="POST /api/trace",
                       payload=lo, resp=r2, elapsed_s=e2, budget_s=BUDGET_TRACE_S)
    if b1 is None or b2 is None:
        return
    p1, p2 = b1.get("power_w"), b2.get("power_w")
    if p1 and p1 > 1e-9:
        ratio = p2 / p1
        if abs(ratio - 0.9) > 1e-4:
            add_wrong(
                findings, cid2, "monotonic", "POST /api/trace", "monotonic",
                f"reflectance 0.9 should collect exactly 0.9x reflectance 1.0's power "
                f"(1.0 -> {p1} W, 0.9 -> {p2} W, ratio {ratio:.6f})",
                {"reflectance_1.0": hi, "reflectance_0.9": lo},
            )


def _check_aperture(caller, findings, rng):
    variant = pick(rng, ["axicon", "cylinder"])
    el = pick(rng, [20.0, 45.0, 70.0])
    if variant == "axicon":
        small = {
            "design": RECT_DESIGN, "mode": "ultra_fast", "optics": "axicon",
            "solar_az_deg": 180.0, "solar_el_deg": el,
            "optics_params": {"aperture_radius_mm": 3000.0},
        }
        big = copy.deepcopy(small)
        big["optics_params"] = {"aperture_radius_mm": 20000.0}
    else:
        small = {
            "design": RECT_DESIGN, "mode": "ultra_fast", "optics": "prime_focus",
            "solar_az_deg": 180.0, "solar_el_deg": el,
            "optics_params": {"receiver_type": "cylinder", "cylinder_radius_mm": 500.0, "cylinder_height_mm": 800.0},
        }
        big = copy.deepcopy(small)
        big["optics_params"] = {"receiver_type": "cylinder", "cylinder_radius_mm": 500.0, "cylinder_height_mm": 20000.0}

    cid = findings.next_case_id()
    r1, e1 = caller.call("POST", "/api/trace", small)
    b1 = judge_common(findings, case_id=cid, area="monotonic", endpoint="POST /api/trace",
                       payload=small, resp=r1, elapsed_s=e1, budget_s=BUDGET_TRACE_S)
    cid2 = findings.next_case_id()
    r2, e2 = caller.call("POST", "/api/trace", big)
    b2 = judge_common(findings, case_id=cid2, area="monotonic", endpoint="POST /api/trace",
                       payload=big, resp=r2, elapsed_s=e2, budget_s=BUDGET_TRACE_S)
    if b1 is None or b2 is None:
        return
    p1, p2 = b1.get("power_w"), b2.get("power_w")
    if p1 is not None and p2 is not None and p2 < p1 - 1e-6 * max(1.0, p1):
        add_wrong(
            findings, cid2, "monotonic", "POST /api/trace", "monotonic",
            f"a larger {variant} aperture/receiver collected LESS power ({p1} W -> {p2} W)",
            {"small": small, "big": big},
        )


def _check_field_count(caller, findings, rng):
    design = mk_rect_design(rng)
    el = pick(rng, [30.0, 60.0])
    counts = [1, 2, 5]
    powers = []
    payloads = []
    for n in counts:
        payload = {
            "design": design, "mode": "ultra_fast", "optics": "prime_focus",
            "solar_az_deg": 180.0, "solar_el_deg": el,
            # Wide ring, generous spacing -- sparse enough that adding a
            # heliostat should not shade an existing one.
            "layout": {"type": "fermat", "n": n, "a_m": 40.0, "r_min_m": 100.0},
        }
        cid = findings.next_case_id()
        resp, elapsed = caller.call("POST", "/api/field/trace", payload)
        body = judge_common(findings, case_id=cid, area="monotonic", endpoint="POST /api/field/trace",
                             payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_TRACE_S)
        payloads.append(payload)
        powers.append(body.get("power_w") if body else None)
    if all(p is not None for p in powers):
        for i in range(1, len(powers)):
            if powers[i] < powers[i - 1] - 1e-6 * max(1.0, powers[i - 1]):
                add_wrong(
                    findings, findings.next_case_id(), "monotonic", "POST /api/field/trace", "monotonic",
                    f"a sparse field of {counts[i]} heliostats collected less power ({powers[i]} W) "
                    f"than {counts[i - 1]} ({powers[i - 1]} W)",
                    {"n": counts, "power_w": powers, "payloads": payloads},
                )


# -- library / setups round trips ---------------------------------------------


def _json_equal(a, b) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def run_round_trip(caller, findings, rng, n):
    for i in range(n):
        kind = pick(rng, ["design", "receiver", "project", "setup"])
        name = f"stress-{kind}-{i}-{rng.randrange(1_000_000)}"
        if kind == "design":
            doc = mk_any_design(rng) if rng.random() < 0.7 else mk_custom_design(rng)
            _round_trip_library(caller, findings, rng, "designs", name, doc)
        elif kind == "receiver":
            optics, params = mk_optics(rng)
            doc = {"optics": optics, "params": params}
            _round_trip_library(caller, findings, rng, "receivers", name, doc)
        elif kind == "project":
            optics, params = mk_optics(rng)
            doc = {
                "schema_version": 1,
                "design": mk_any_design(rng),
                "receiver": {"optics": optics, "params": params},
                "field": {"layout": mk_layout(rng, max_n=10)},
                "sun": {"azimuth_deg": pick(rng, SUN_AZIMUTHS), "elevation_deg": pick(rng, [10.0, 45.0, 89.0])},
                "run": {"mode": pick(rng, MODES)},
            }
            _round_trip_library(caller, findings, rng, "projects", name, doc)
        else:
            doc = {"free_form": True, "n": rng.randint(0, 1000), "nested": {"a": [1, 2, mk_rect_design(rng)]}}
            _round_trip_setup(caller, findings, rng, name, doc)


def _round_trip_library(caller, findings, rng, collection, name, doc):
    cid = findings.next_case_id()
    save_payload = {"name": name, "document": doc}
    r1, e1 = caller.call("POST", f"/api/library/{collection}", save_payload)
    body1 = judge_common(findings, case_id=cid, area="round_trip", endpoint=f"POST /api/library/{collection}",
                          payload=save_payload, resp=r1, elapsed_s=e1, budget_s=BUDGET_GEOMETRY_S)
    if body1 is None:
        return  # a document this generator built that the schema itself rejects is not a stress finding

    cid2 = findings.next_case_id()
    r2, e2 = caller.call("GET", f"/api/library/{collection}/{name}")
    body2 = judge_common(findings, case_id=cid2, area="round_trip", endpoint=f"GET /api/library/{collection}/{{name}}",
                          payload={"collection": collection, "name": name}, resp=r2, elapsed_s=e2, budget_s=BUDGET_GEOMETRY_S)
    if body2 is None:
        add_wrong(
            findings, cid2, "round_trip", f"GET /api/library/{collection}/{{name}}", "roundtrip",
            f"saved {collection}/{name!r} but loading it back failed", save_payload,
        )
        return
    loaded = body2.get("document")
    if not _json_equal(loaded, doc):
        add_wrong(
            findings, cid2, "round_trip", f"GET /api/library/{collection}/{{name}}", "roundtrip",
            f"{collection}/{name!r} round-trip changed the document",
            save_payload, extra={"saved": doc, "loaded": loaded},
        )

    caller.call("DELETE", f"/api/library/{collection}/{name}")


def _round_trip_setup(caller, findings, rng, name, doc):
    cid = findings.next_case_id()
    save_payload = {"name": name, "document": doc}
    r1, e1 = caller.call("POST", "/api/setups", save_payload)
    body1 = judge_common(findings, case_id=cid, area="round_trip", endpoint="POST /api/setups",
                          payload=save_payload, resp=r1, elapsed_s=e1, budget_s=BUDGET_GEOMETRY_S)
    if body1 is None:
        return
    cid2 = findings.next_case_id()
    r2, e2 = caller.call("GET", f"/api/setups/{name}")
    body2 = judge_common(findings, case_id=cid2, area="round_trip", endpoint="GET /api/setups/{name}",
                          payload={"name": name}, resp=r2, elapsed_s=e2, budget_s=BUDGET_GEOMETRY_S)
    if body2 is None:
        add_wrong(findings, cid2, "round_trip", "GET /api/setups/{name}", "roundtrip",
                  f"saved setup {name!r} but loading it back failed", save_payload)
        return
    if not _json_equal(body2.get("document"), doc):
        add_wrong(
            findings, cid2, "round_trip", "GET /api/setups/{name}", "roundtrip",
            f"setup {name!r} round-trip changed the document", save_payload,
            extra={"saved": doc, "loaded": body2.get("document")},
        )
    caller.call("DELETE", f"/api/setups/{name}")


# -- day sweep (kept tiny: 1-2 heliostats, coarse step) ------------------------


def run_day_sweep(caller, findings, rng, n):
    for _ in range(n):
        cid = findings.next_case_id()
        optics, params = mk_optics(rng)
        payload = {
            "design": RECT_DESIGN,
            "mode": "ultra_fast",
            "optics": optics,
            "optics_params": params,
            "solar_az_deg": 180.0,
            "solar_el_deg": 45.0,  # ignored by the endpoint; the sun moves
            "site": {
                "latitude_deg": pick(rng, [-10.0, 0.0, 40.0]),
                "month": pick(rng, [3, 6, 12]),
                "day": pick(rng, [1, 21]),
            },
            "hour_step": pick(rng, [3.0, 6.0]),
            "layout": {"type": "fermat", "n": pick(rng, [1, 2])},
        }
        resp, elapsed = caller.call("POST", "/api/day/start", payload)
        body = judge_common(findings, case_id=cid, area="day_sweep", endpoint="POST /api/day/start",
                             payload=payload, resp=resp, elapsed_s=elapsed, budget_s=BUDGET_GEOMETRY_S)
        if body is None or "job_id" not in body:
            continue
        job_id = body["job_id"]

        t0 = time.perf_counter()
        deadline = t0 + 120.0
        result = None
        n_steps_hint = None
        while time.perf_counter() < deadline:
            r, _ = caller.call("GET", f"/api/day/status/{job_id}")
            snap = r.json() if r.status_code == 200 else {}
            n_steps_hint = snap.get("total", n_steps_hint)
            if snap.get("state") in ("done", "error", "cancelled"):
                result = snap
                break
            time.sleep(0.05)
        elapsed_total = time.perf_counter() - t0

        if result is None:
            add_wrong(
                findings, cid, "day_sweep", "GET /api/day/status/{job_id}", "timing",
                f"day sweep did not finish within 120s (n_heliostats={payload['layout']['n']}, "
                f"hour_step={payload['hour_step']})",
                payload, elapsed_total,
            )
            continue
        if result.get("state") == "error":
            add_wrong(
                findings, cid, "day_sweep", "GET /api/day/status/{job_id}", "crash",
                f"day sweep job errored: {result.get('error')}", payload,
            )
            continue

        cid2 = findings.next_case_id()
        r3, e3 = caller.call("GET", f"/api/day/result/{job_id}")
        body3 = judge_common(findings, case_id=cid2, area="day_sweep", endpoint="GET /api/day/result/{job_id}",
                              payload={"job_id": job_id}, resp=r3, elapsed_s=e3, budget_s=None)
        if body3 is not None:
            steps = body3.get("steps") or []
            if n_steps_hint and steps and elapsed_total / max(1, len(steps)) > BUDGET_DAY_STEP_S:
                add_wrong(
                    findings, cid2, "day_sweep", "POST /api/day/start", "timing",
                    f"{len(steps)} timesteps took {elapsed_total:.1f}s total "
                    f"({elapsed_total / len(steps):.1f}s/step, budget {BUDGET_DAY_STEP_S:.0f}s/step)",
                    payload, elapsed_total,
                )
            energy = body3.get("energy_kwh")
            if energy is not None and energy < -1e-6:
                add_wrong(findings, cid2, "day_sweep", "GET /api/day/result/{job_id}", "physics",
                          f"energy_kwh={energy} < 0", payload)


# ---------------------------------------------------------------------------
# report


def print_report(findings: Findings, elapsed_total: float) -> None:
    by_sev = findings.by_severity()
    print()
    print("=" * 88)
    print("STRESS TEST REPORT")
    print("=" * 88)
    print(f"cases run: {findings.cases_run}   http calls made: {findings.calls_made}   "
          f"wall time: {elapsed_total:.1f}s")
    counts = {sev: len(items) for sev, items in by_sev.items()}
    print(f"findings: {sum(counts.values())}  ->  " + ", ".join(
        f"{sev}={counts.get(sev, 0)}" for sev in ("crash", "hang", "wrong", "other")
    ))
    print()

    if not findings.items:
        print("No findings. Every case returned a well-formed response inside its time "
              "budget, with finite numbers and no rule violation this harness checks for.")
        return

    for sev in ("crash", "hang", "wrong", "other"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        print("=" * 88)
        print(f"{sev.upper()} ({len(items)})")
        print("=" * 88)
        by_class: dict[str, list[Finding]] = {}
        for f in items:
            by_class.setdefault(f.fclass, []).append(f)
        for fclass, fs in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            print(f"\n-- {fclass} ({len(fs)}) " + "-" * max(0, 70 - len(fclass)))
            for f in fs:
                print(f"\n[case {f.case_id}] area={f.area} endpoint={f.endpoint}")
                if f.elapsed_s is not None:
                    budget = f"{f.budget_s:.1f}s" if f.budget_s is not None else "n/a"
                    print(f"  elapsed={f.elapsed_s:.3f}s budget={budget}")
                print(f"  {f.message}")
                print("  payload:")
                print(indent(json.dumps(f.payload, indent=2, default=str), "    "))
                if f.extra:
                    print("  extra:")
                    print(indent(json.dumps(f.extra, indent=2, default=str), "    "))


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# main


AREA_ORDER = [
    "design_matrix",
    "custom_polygons",
    "degenerate_optics",
    "sun_positions",
    "field_positions",
    "field_trace",
    "cross_agreement",
    "monotonic",
    "round_trip",
    "day_sweep",
    "caps",
]

# (base-quick-count, full-multiplier) per generator-driven area; "caps" runs
# a fixed set of cases regardless of scale.
BASE_COUNTS = {
    "design_matrix": (70, 10),
    "custom_polygons": (30, 8),
    "degenerate_optics": (35, 8),
    "sun_positions": (30, 8),
    "field_positions": (25, 8),
    "field_trace": (20, 8),
    "cross_agreement": (15, 8),
    "monotonic": (18, 6),
    "round_trip": (25, 8),
    "day_sweep": (5, 3),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    depth = ap.add_mutually_exclusive_group()
    depth.add_argument("--quick", action="store_true", help="a few hundred cases (default)")
    depth.add_argument("--full", action="store_true", help="thousands of cases")
    ap.add_argument("--focus", choices=AREA_ORDER, default=None, help="hammer one area only")
    ap.add_argument("--seed", type=int, default=20260825, help="base seed for deterministic sampling")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    full = bool(args.full)

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    caller = Caller(client)
    findings = Findings()

    runners: dict[str, Callable] = {
        "design_matrix": run_design_matrix,
        "custom_polygons": run_custom_polygons,
        "degenerate_optics": run_degenerate_optics,
        "sun_positions": run_sun_positions,
        "field_positions": run_field_positions,
        "field_trace": run_field_trace_general,
        "cross_agreement": run_cross_agreement,
        "monotonic": run_monotonic,
        "round_trip": run_round_trip,
        "day_sweep": run_day_sweep,
    }

    areas = [args.focus] if args.focus and args.focus != "caps" else (
        [args.focus] if args.focus == "caps" else AREA_ORDER
    )

    t0 = time.perf_counter()
    print(f"seed={args.seed} mode={'full' if full else 'quick'} "
          f"focus={args.focus or 'all'} library_dir={os.environ['HELIOSTAT_LIBRARY_DIR']}", flush=True)

    for i, area in enumerate(areas):
        t_area = time.perf_counter()
        if area == "caps":
            print(f"[{area}] running fixed cap cases...", flush=True)
            rng = random.Random((args.seed, area))
            run_caps(caller, findings, rng)
            findings.cases_run += 4
            print(f"[{area}] done in {time.perf_counter() - t_area:.1f}s", flush=True)
            continue
        base, mult = BASE_COUNTS[area]
        n = base * (mult if full else 1)
        if args.focus:
            n = max(n, base * (mult if full else 3))
        rng = random.Random((args.seed, area))
        print(f"[{area}] running {n} cases...", flush=True)
        runners[area](caller, findings, rng, n)
        findings.cases_run += n
        print(f"[{area}] done in {time.perf_counter() - t_area:.1f}s "
              f"({(time.perf_counter() - t_area) / max(1, n):.3f}s/case)", flush=True)

    elapsed_total = time.perf_counter() - t0
    print_report(findings, elapsed_total)

    crashes = len(findings.by_severity().get("crash", []))
    return 1 if crashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
