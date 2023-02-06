"""
This script demonstrates loading the bolometer sensors and the MAST-U
vessel and plotting the lines of sight. A single ray is used for each
LOS.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyuda

from raysect.optical import World
from raysect.optical.material.absorber import AbsorbingSurface

from cherab.mastu.machine import import_mastu_mesh
from cherab.mastu.bolometry import load_default_bolometer_config


world = World()
sxd_outer = load_default_bolometer_config('SXDL - Outer', parent=world)
sxd_upper = load_default_bolometer_config('SXDL - Upper', parent=world)
sxdu_outer = load_default_bolometer_config('SXDU - Outer', parent=world)
core_poloidal = load_default_bolometer_config('CORE - Poloidal', parent=world)
# Don't plot tangential array on a poloidal cross section
detectors = list(sxd_outer) + list(sxd_upper) + list(sxdu_outer) + list(core_poloidal)

import_mastu_mesh(world, override_material=AbsorbingSurface())

client = pyuda.Client()
passive_structures = client.geometry('/passive/efit', 50000)


fig, ax = plt.subplots()
for detector in detectors:
    print('Tracing detector {}'.format(detector.name))
    start, end, _ = detector.trace_sightline()
    start_r = np.hypot(start.x, start.y)
    end_r = np.hypot(end.x, end.y)
    ax.plot([start_r, end_r], [start.z, end.z], 'r')

passive_structures.plot(ax_2d=ax, color='k')
ax.axis('equal')
plt.show()
