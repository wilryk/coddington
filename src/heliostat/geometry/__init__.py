"""Geometric building blocks: heliostats, receivers, secondary optics."""

from .heliostat import heliostat_orientation, heliostat_shape
from .receiver import CylinderReceiver, FlatWindowReceiver, FrustumReceiver, Receiver
from .secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    PyramidSecondary,
    Secondary,
)
from .shading import (
    MirrorGeometry,
    SecondaryCone,
    SecondaryDisc,
    build_geometries,
    occlusion_efficiency,
    search_radius_for,
    shading_blocking,
    sun_vector,
)

__all__ = [
    "heliostat_orientation",
    "heliostat_shape",
    "Receiver",
    "FlatWindowReceiver",
    "CylinderReceiver",
    "FrustumReceiver",
    "Secondary",
    "NoSecondary",
    "AxiconSecondary",
    "CassegrainSecondary",
    "PyramidSecondary",
    "MirrorGeometry",
    "SecondaryCone",
    "SecondaryDisc",
    "build_geometries",
    "occlusion_efficiency",
    "search_radius_for",
    "shading_blocking",
    "sun_vector",
]
