"""
This script demonstrates how to calculate the etendue of each bolometer
foil. In this case, loading the vessel is unnecessary: the etendue is
determined by the size and orientation of the foils and slits.

The description of the foils is taken from the 'as-drawn' data in the
MAST-U Machine Description. A sufficiently high ray count is needed to
obtain good statistics.
"""
import sys
import numpy as np
from raysect.optical import World

from cherab.mastu.bolometry import load_default_bolometer_config


world = World()


try:
    camera = sys.argv[1]
except (IndexError, ValueError):
    print('Usage: {} <camera>'.format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

if camera == 'Outer':
    camera_name = 'SXDL - Outer'
elif camera == 'Upper':
    camera_name = 'SXDL - Upper'
elif camera == 'Poloidal':
    camera_name = 'CORE - Poloidal'
elif camera == 'Tangential':
    camera_name = 'CORE - Tangential'
elif camera == 'OuterSXDU':
    camera_name = 'SXDU - Outer'
else:
    raise ValueError('Camera should be "Outer", "Upper", "Poloidal", "Tangential", "OuterSXDU"')

shot = 50000
bolometer_camera = load_default_bolometer_config(camera_name, parent=world, shot=shot)
etendues = np.empty((len(bolometer_camera), 2))
for i, detector in enumerate(bolometer_camera):
    etendue, etendue_error = detector.calculate_etendue(ray_count=250000)
    print('Etendue for {}: {} +- {} m**2.sr'.format(detector.name, etendue, etendue_error))
    etendues[i, :] = etendue, etendue_error
np.save('{}_etendue'.format(camera.lower()), etendues)
