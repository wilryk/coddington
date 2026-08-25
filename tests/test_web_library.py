"""HTTP-level gate for the Library (``/api/library/{collection}``) --
designs, receivers and projects (docs/ui-spec.md 5), plus the parity check
that keeps heliostat.web.builtin_library's numbers equal to app.py's own
optics defaults.

Split out of ``test_web.py`` (already long), same reasoning as
``test_web_geometry.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.web.app import (  # noqa: E402
    _OPTICS_PARAM_MODELS,
    RectParams,
    create_app,
)
from heliostat.web.builtin_library import BUILTIN_DESIGNS, BUILTIN_RECEIVERS  # noqa: E402

RECT_DOC = {"type": "rect", "width_mm": 1000.0, "height_mm": 800.0}
BUILTIN_DESIGN_NAMES = list(BUILTIN_DESIGNS)
BUILTIN_RECEIVER_NAMES = list(BUILTIN_RECEIVERS)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture
def library_dir(tmp_path, monkeypatch):
    """Point the library store at a temp dir, never the real home directory."""
    monkeypatch.setenv("HELIOSTAT_LIBRARY_DIR", str(tmp_path / "library"))
    return tmp_path / "library"


def _project_document(**overrides):
    doc = {
        "schema_version": 1,
        "design": RECT_DOC,
        "receiver": {"optics": "prime_focus", "params": {}},
        "field": {"layout": {"type": "fermat", "n": 5}},
        "sun": {"azimuth_deg": 180.0, "elevation_deg": 45.0},
        "run": {"mode": "ultra_fast"},
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# built-ins: always present, never editable


@pytest.mark.parametrize(
    "collection,names",
    [("designs", BUILTIN_DESIGN_NAMES), ("receivers", BUILTIN_RECEIVER_NAMES)],
)
def test_builtins_are_listed_first(client, library_dir, collection, names):
    listed = client.get(f"/api/library/{collection}").json()["entries"]
    assert [e["name"] for e in listed[: len(names)]] == names
    assert all(e["builtin"] for e in listed[: len(names)])


@pytest.mark.parametrize("name", BUILTIN_RECEIVER_NAMES)
def test_builtin_receiver_loads_from_the_constant(client, library_dir, name):
    resp = client.get(f"/api/library/receivers/{name}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["builtin"] is True
    assert data["document"] == BUILTIN_RECEIVERS[name]


@pytest.mark.parametrize("name", BUILTIN_DESIGN_NAMES)
def test_builtin_design_loads_from_the_constant(client, library_dir, name):
    resp = client.get(f"/api/library/designs/{name}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["builtin"] is True
    assert data["document"] == BUILTIN_DESIGNS[name]


@pytest.mark.parametrize(
    "collection,name",
    [("designs", BUILTIN_DESIGN_NAMES[0]), ("receivers", BUILTIN_RECEIVER_NAMES[0])],
)
def test_builtins_cannot_be_overwritten(client, library_dir, collection, name):
    resp = client.post(
        f"/api/library/{collection}",
        json={"name": name, "document": {"type": "rect", "width_mm": 1, "height_mm": 1}},
    )
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "collection,name",
    [("designs", BUILTIN_DESIGN_NAMES[0]), ("receivers", BUILTIN_RECEIVER_NAMES[0])],
)
def test_builtins_cannot_be_deleted(client, library_dir, collection, name):
    resp = client.delete(f"/api/library/{collection}/{name}")
    assert resp.status_code == 409
    # Still there afterwards.
    assert client.get(f"/api/library/{collection}/{name}").status_code == 200


# ---------------------------------------------------------------------------
# parity: the built-in numbers cannot silently drift from app.py's defaults


@pytest.mark.parametrize("name", BUILTIN_RECEIVER_NAMES)
def test_builtin_receiver_params_validate_and_equal_the_models_own_defaults(name):
    doc = BUILTIN_RECEIVERS[name]
    model = _OPTICS_PARAM_MODELS[doc["optics"]]
    validated = model.model_validate(doc["params"])
    assert validated == model()  # the model's own defaults, unmodified


@pytest.mark.parametrize("name", BUILTIN_DESIGN_NAMES)
def test_builtin_design_validates_as_a_rect_param(name):
    doc = BUILTIN_DESIGNS[name]
    validated = RectParams.model_validate(doc)
    assert validated.width_mm == 5000.0
    assert validated.height_mm == 3000.0


# ---------------------------------------------------------------------------
# user entries: save, list, load, delete


def test_design_round_trips(client, library_dir):
    saved = client.post("/api/library/designs", json={"name": "My rect", "document": RECT_DOC})
    assert saved.status_code == 200
    assert saved.json()["name"] == "My rect"

    listed = client.get("/api/library/designs").json()["entries"]
    user_entries = [e for e in listed if not e["builtin"]]
    assert [e["name"] for e in user_entries] == ["My rect"]
    assert "saved_at" in user_entries[0]

    loaded = client.get("/api/library/designs/My rect").json()
    assert loaded["document"] == RECT_DOC
    assert loaded["builtin"] is False

    assert client.delete("/api/library/designs/My rect").status_code == 200
    listed_after = client.get("/api/library/designs").json()["entries"]
    assert all(e["builtin"] for e in listed_after)


def test_receiver_round_trips(client, library_dir):
    doc = {"optics": "axicon", "params": {"apex_height_mm": 20000.0}}
    saved = client.post("/api/library/receivers", json={"name": "My tower", "document": doc})
    assert saved.status_code == 200
    loaded = client.get("/api/library/receivers/My tower").json()
    assert loaded["document"] == doc


def test_saving_the_same_name_twice_overwrites(client, library_dir):
    client.post("/api/library/designs", json={"name": "dup", "document": RECT_DOC})
    client.post(
        "/api/library/designs",
        json={"name": "dup", "document": {**RECT_DOC, "width_mm": 2000.0}},
    )
    loaded = client.get("/api/library/designs/dup").json()
    assert loaded["document"]["width_mm"] == 2000.0
    listed = [e for e in client.get("/api/library/designs").json()["entries"] if not e["builtin"]]
    assert len(listed) == 1


def test_loading_missing_entry_is_404(client, library_dir):
    resp = client.get("/api/library/designs/does-not-exist")
    assert resp.status_code == 404


def test_deleting_missing_entry_is_404(client, library_dir):
    resp = client.delete("/api/library/designs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.parametrize("name", ["../escape", "sub/dir", ".hidden", "CON", "x" * 65])
def test_unsafe_names_are_refused(client, library_dir, name):
    resp = client.post("/api/library/designs", json={"name": name, "document": RECT_DOC})
    assert resp.status_code == 422


def test_unreadable_entry_file_is_skipped_not_fatal(client, library_dir):
    client.post("/api/library/designs", json={"name": "good", "document": RECT_DOC})
    (library_dir / "designs").mkdir(parents=True, exist_ok=True)
    (library_dir / "designs" / "broken.json").write_text("{not json", encoding="utf-8")
    listed = [e for e in client.get("/api/library/designs").json()["entries"] if not e["builtin"]]
    assert [e["name"] for e in listed] == ["good"]


# ---------------------------------------------------------------------------
# document validation per collection


def test_invalid_design_document_is_422(client, library_dir):
    resp = client.post(
        "/api/library/designs",
        json={"name": "bad", "document": {"type": "rect", "width_mm": -1, "height_mm": 1}},
    )
    assert resp.status_code == 422
    assert "width_mm" in resp.json()["detail"]


def test_unknown_design_type_is_422(client, library_dir):
    resp = client.post(
        "/api/library/designs", json={"name": "bad", "document": {"type": "hexagon"}}
    )
    assert resp.status_code == 422


def test_invalid_receiver_document_is_422(client, library_dir):
    doc = {"optics": "prime_focus", "params": {"focus_height_mm": -1}}
    resp = client.post("/api/library/receivers", json={"name": "bad", "document": doc})
    assert resp.status_code == 422


def test_unknown_receiver_optics_is_422(client, library_dir):
    resp = client.post(
        "/api/library/receivers",
        json={"name": "bad", "document": {"optics": "parabolic_dish", "params": {}}},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# unknown collection


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/library/bogus"),
        ("get", "/api/library/bogus/name"),
        ("delete", "/api/library/bogus/name"),
    ],
)
def test_unknown_collection_is_404(client, library_dir, method, path):
    resp = getattr(client, method)(path)
    assert resp.status_code == 404


def test_unknown_collection_post_is_404(client, library_dir):
    resp = client.post("/api/library/bogus", json={"name": "x", "document": {}})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# projects and the v1 schema


def test_project_round_trips(client, library_dir):
    doc = _project_document()
    saved = client.post("/api/library/projects", json={"name": "proj1", "document": doc})
    assert saved.status_code == 200

    loaded = client.get("/api/library/projects/proj1").json()
    assert loaded["document"] == doc
    assert loaded["builtin"] is False


def test_projects_have_no_builtins(client, library_dir):
    listed = client.get("/api/library/projects").json()["entries"]
    assert listed == []  # nothing yet -- and no built-in entries at all


def test_project_missing_schema_version_is_422(client, library_dir):
    doc = _project_document()
    del doc["schema_version"]
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 422


def test_project_wrong_schema_version_is_422(client, library_dir):
    doc = _project_document(schema_version=3)
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 422
    assert "schema_version" in resp.json()["detail"]


def test_project_v1_document_still_loads(client, library_dir):
    """A document saved before the ``runs`` field existed must still open --
    schema_version 1 carries no ``runs`` key at all, not an empty one."""
    doc = _project_document()
    assert "runs" not in doc
    saved = client.post("/api/library/projects", json={"name": "old", "document": doc})
    assert saved.status_code == 200

    loaded = client.get("/api/library/projects/old").json()
    assert loaded["document"] == doc


def test_project_v2_document_carries_its_saved_runs(client, library_dir):
    doc = _project_document(schema_version=2, runs=["day-2026-03-21-1"])
    saved = client.post("/api/library/projects", json={"name": "p2", "document": doc})
    assert saved.status_code == 200
    loaded = client.get("/api/library/projects/p2").json()
    assert loaded["document"]["runs"] == ["day-2026-03-21-1"]


def test_project_bad_receiver_params_is_422(client, library_dir):
    doc = _project_document(receiver={"optics": "prime_focus", "params": {"focus_height_mm": -5}})
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 422


def test_project_bad_design_is_422(client, library_dir):
    doc = _project_document(design={"type": "rect", "width_mm": -1, "height_mm": 1})
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 422


def test_project_extra_field_is_422(client, library_dir):
    """extra="forbid" -- a typo'd or future-version field must not be
    silently accepted and then silently ignored."""
    doc = _project_document()
    doc["bogus_extra_field"] = True
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 422


def test_project_single_heliostat_field_is_valid(client, library_dir):
    doc = _project_document(field={"heliostat_x_mm": 0.0, "heliostat_y_mm": -90000.0})
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 200


def test_project_site_block_is_optional(client, library_dir):
    doc = _project_document(
        sun={
            "azimuth_deg": 180.0,
            "elevation_deg": 45.0,
            "site": {"latitude_deg": -10.0, "longitude_deg": -52.0, "timezone_h": -3.0},
        }
    )
    resp = client.post("/api/library/projects", json={"name": "p", "document": doc})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# saved runs (docs/ui-spec.md 4: "runs save with the project")


def _run_document(**overrides):
    doc = {
        "kind": "day",
        "project_name": None,
        "request": {
            "design": RECT_DOC,
            "mode": "ultra_fast",
            "optics": "prime_focus",
            "solar_az_deg": 180.0,
            "solar_el_deg": 45.0,
        },
        "result": {"date": "2026-03-21", "energy_kwh": 12.3, "steps": []},
        "flux_pngs": {},
    }
    doc.update(overrides)
    return doc


def test_runs_have_no_builtins(client, library_dir):
    assert client.get("/api/library/runs").json()["entries"] == []


def test_run_round_trips(client, library_dir):
    doc = _run_document()
    saved = client.post("/api/library/runs", json={"name": "day-2026-03-21", "document": doc})
    assert saved.status_code == 200

    listed = client.get("/api/library/runs").json()["entries"]
    assert [e["name"] for e in listed] == ["day-2026-03-21"]
    assert listed[0]["builtin"] is False
    assert listed[0]["size_bytes"] > 0

    loaded = client.get("/api/library/runs/day-2026-03-21").json()
    assert loaded["document"] == doc
    assert loaded["builtin"] is False

    assert client.delete("/api/library/runs/day-2026-03-21").status_code == 200
    assert client.get("/api/library/runs").json()["entries"] == []


def test_run_disk_footprint_is_the_real_file_size(client, library_dir):
    """The Manage-saved-runs footprint must be the truth, not an estimate --
    check it against the actual bytes on disk for that one entry."""
    doc = _run_document(flux_pngs={"0": "not-really-a-png-but-takes-up-space" * 50})
    client.post("/api/library/runs", json={"name": "heavy", "document": doc})

    reported = client.get("/api/library/runs").json()["entries"][0]["size_bytes"]
    on_disk = (library_dir / "runs" / "heavy.json").stat().st_size
    assert reported == on_disk


def test_run_kind_year_round_trips(client, library_dir):
    doc = _run_document(
        kind="year",
        request={
            "design": RECT_DOC,
            "mode": "ultra_fast",
            "optics": "prime_focus",
            "solar_az_deg": 180.0,
            "solar_el_deg": 45.0,
            "fast_mode": True,
        },
        result={"annual_energy_mwh": 1.2, "days": []},
    )
    saved = client.post("/api/library/runs", json={"name": "year-2026", "document": doc})
    assert saved.status_code == 200
    loaded = client.get("/api/library/runs/year-2026").json()
    assert loaded["document"]["kind"] == "year"


def test_run_unknown_kind_is_422(client, library_dir):
    doc = _run_document(kind="month")
    resp = client.post("/api/library/runs", json={"name": "bad", "document": doc})
    assert resp.status_code == 422


def test_run_missing_request_is_422(client, library_dir):
    doc = _run_document()
    del doc["request"]
    resp = client.post("/api/library/runs", json={"name": "bad", "document": doc})
    assert resp.status_code == 422


def test_run_can_be_tagged_with_a_project(client, library_dir):
    doc = _run_document(project_name="p2")
    client.post("/api/library/runs", json={"name": "tagged", "document": doc})
    loaded = client.get("/api/library/runs/tagged").json()
    assert loaded["document"]["project_name"] == "p2"


def test_run_names_cannot_collide_with_a_builtin_since_there_are_none(client, library_dir):
    # runs has no built-ins (unlike designs/receivers), so any safe name saves.
    resp = client.post("/api/library/runs", json={"name": "anything", "document": _run_document()})
    assert resp.status_code == 200
