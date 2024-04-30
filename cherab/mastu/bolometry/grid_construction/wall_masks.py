"""
This module contains the masks used to select the cells from a 2D grid
to be used in the reconstruction grid.

TODO: this should not reproduce functionality available in
cherab.mastu.machine, once issue #1 is closed.
"""
import numpy as np
import matplotlib.pyplot as plt
import pyuda

from cherab.core.math import PolygonMask2D

client = pyuda.Client()


_MASTU_SXD_GRID_POLYGON = np.array([
    (0.84, -1.551),
    (0.864, -1.573),
    (0.893, -1.587),
    (0.925, -1.59),
    (1.69, -1.552),
    (1.73, -1.68),
    (1.35, -2.06),
    (1.09, -2.06),
    (0.9, -1.87),
    (0.63, -1.6),
])


SXD_POLYGON_MASK = PolygonMask2D(_MASTU_SXD_GRID_POLYGON)

# Use as the core mask the part of the limiting surface with
# z_max,sxd < z < -z_max,sxd,
# where z_max_sxd is the least negative Z value from the SXD polygon
_LIMITING_SURFACE = client.geometry('/limiter/efit', 50000)
_MASTU_CORE_GRID_POLYGON = np.array([
    _LIMITING_SURFACE.data.R,
    _LIMITING_SURFACE.data.Z
]).T
_MIN_CORE_Z = _MASTU_SXD_GRID_POLYGON[:, 1].max()
_NON_SXD_LIMITING_SURFACE = (
    (_MIN_CORE_Z < _MASTU_CORE_GRID_POLYGON[:, 1])
    & (_MASTU_CORE_GRID_POLYGON[:, 1] < -_MIN_CORE_Z)
)
_MASTU_CORE_GRID_POLYGON = _MASTU_CORE_GRID_POLYGON[_NON_SXD_LIMITING_SURFACE]

CORE_POLYGON_MASK = PolygonMask2D(_MASTU_CORE_GRID_POLYGON)


if __name__ == '__main__':

    plt.ion()
    plt.figure()
    plt.plot(_MASTU_SXD_GRID_POLYGON[:, 0], _MASTU_SXD_GRID_POLYGON[:, 1], 'k')
    plt.axis('equal')
