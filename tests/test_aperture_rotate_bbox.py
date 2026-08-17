"""Regression: Rotate.bbox() must rotate the child's box FORWARD.

A rect offset to +u, rotated +90 degrees CCW, must end up at +v. The
original code applied the inverse rotation (landing it at -v) — invisible
for centre-symmetric shapes and full circular arrays, corrupting for any
individually-rotated off-centre shape, which is exactly what flower
petals are. Found during design-layer work; contains() was always right,
only the box was mirrored.
"""

from heliostat.geometry.aperture import Rect


def test_rotate_bbox_forward_rotation_of_offcentre_shape():
    shape = Rect(200.0, 100.0).translated(1000.0, 0.0).rotated(90.0)
    u0, u1, v0, v1 = shape.bbox()
    assert 800.0 < v0 < v1 < 1200.0, (u0, u1, v0, v1)
    assert abs(u0) < 200.0 and abs(u1) < 200.0, (u0, u1)
    assert shape.contains(0.0, 1000.0)
    assert not shape.contains(0.0, -1000.0)
    # The numeric area integrates over bbox; with the fixed box it matches.
    assert abs(shape.area_mm2(1024) - 200.0 * 100.0) / (200.0 * 100.0) < 5e-3
