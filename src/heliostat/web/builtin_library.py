"""Built-in library entries: the manuscript's own tower geometries and
heliostat design, read-only and always present (docs/ui-spec.md 5).

Plain dict constants rather than the pydantic models :mod:`heliostat.web.app`
validates them against: this module has to be importable from
:mod:`heliostat.web.library`'s side of things without pulling in FastAPI/
pydantic, and a built-in is data, not behaviour. ``tests/test_web_library.py``
is the parity gate -- it validates every entry here through the matching
model in ``app.py`` *and* asserts the numbers equal that model's own
defaults, so this file and the app's module constants cannot silently drift
apart.

Numbers are copied from ``heliostat.web.app``'s optics defaults
(``PRIME_FOCUS_HEIGHT_MM`` and its neighbours), which are themselves copied
from ``examples/paper/reproduce.py`` -- see that module's docstring for
where each one comes from. Copied rather than imported for the same reason
``app._geometry_for`` does not import ``_geometry_for`` from the test suite:
a built-in library entry is content, not a physics dependency, and neither
script is a stable import surface for the other.
"""

from __future__ import annotations

#: Receiver window half-extent, matching app.WINDOW_MM -- every built-in
#: receiver uses the library's own default window, since none of the three
#: manuscript layouts calls for a different one.
_WINDOW_MM = 2000.0

#: Receiver configs, keyed by display name. Each document is the shape
#: app.ReceiverDocument validates: which optics layout, and the params that
#: resolve against that layout's own model.
BUILTIN_RECEIVERS: dict[str, dict] = {
    "Prime focus 35.3 m": {
        "optics": "prime_focus",
        "params": {
            "focus_height_mm": 35335.0,
            "window_half_u_mm": _WINDOW_MM,
            "window_half_v_mm": _WINDOW_MM,
        },
    },
    "Axicon 27 m / 20 deg / 14 m": {
        "optics": "axicon",
        "params": {
            "apex_height_mm": 27000.0,
            "half_angle_deg": 20.0,
            "aperture_radius_mm": 14000.0,
            "receiver_z_mm": 7000.0,
            "window_half_u_mm": _WINDOW_MM,
            "window_half_v_mm": _WINDOW_MM,
        },
    },
    "Cassegrain relay": {
        "optics": "cassegrain",
        "params": {
            "vertex_z_mm": 26993.999446877,
            "focus_height_mm": 34892.4,
            "receiver_z_mm": 7000.0,
            "aperture_radius_mm": 14000.0,
            "window_half_u_mm": _WINDOW_MM,
            "window_half_v_mm": _WINDOW_MM,
        },
    },
}

#: The manuscript heliostat, 5 x 3 m, one entry per surface figure. Each
#: document is the shape app.DesignParams validates (a RectParams dict).
BUILTIN_DESIGNS: dict[str, dict] = {
    f"Manuscript 5 x 3 m — {surface}": {
        "type": "rect",
        "width_mm": 5000.0,
        "height_mm": 3000.0,
        "surface": surface,
    }
    for surface in ("twisting", "spherical", "flat")
}
