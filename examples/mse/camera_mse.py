
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# Port geometry for the MSE diagnostic, uses a pin hole camera to view attenuated neutral beam emission.
# Plasma is deuterium, with ~1% C6 impurities
#

# External imports
import matplotlib.pyplot as plt
import pickle
import idlbridge as idl
import numpy as np
from scipy.constants import electron_mass, atomic_mass
from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.core.workflow import SerialEngine
from raysect.optical import Ray, d65_white, World, Point3D, Vector3D, translate, rotate
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Sphere, Box, Intersect, Cylinder

import pyuda

# Internal imports
from cherab.core.math import Interpolate1DCubic, IsoMapper2D, IsoMapper3D, AxisymmetricMapper, Blend2D, Constant2D, Constant3D, \
    VectorAxisymmetricMapper
from cherab.core import Plasma, Maxwellian, Species, Beam
from cherab.core.atomic import elements
from cherab.core.atomic import Line, deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.core.model import SingleRayAttenuator, BeamCXLine
from cherab.openadas import OpenADAS

from cherab.mastu.equilibrium import MASTUEquilibrium


plt.ion()

client = pyuda.Client()

PULSE = 45272
TIME = 0.2

world = World()

adas = OpenADAS(permit_extrapolation=True)  # create atomic data source

# ########################### PLASMA EQUILIBRIUM ############################ #
print('Plasma equilibrium')

equilibrium = MASTUEquilibrium(PULSE)
equil_time_slice = equilibrium.time(TIME)
psin_2d = equil_time_slice.psi_normalised
psin_3d = AxisymmetricMapper(equil_time_slice.psi_normalised)
inside_lcfs = equil_time_slice.inside_lcfs

# ########################### PLASMA CONFIGURATION ########################## #
print('Plasma configuration')

plasma = Plasma(parent=world)
plasma.atomic_data = adas
plasma.b_field = VectorAxisymmetricMapper(equil_time_slice.b_field)
plasma_index = equilibrium._find_nearest(equilibrium.time_slices, TIME)

r = equilibrium.r.data #R,Z co-ordinates defined in EFIT
z = equilibrium.z.data

#Find Psi(R,Z=0)
psi2d = np.zeros((len(r),len(z)))

for i in range(len(r)):
    for j in range(len(z)):
        psi2d[j,i] = psin_2d(r[i],z[j])

psi_z0 = psi2d[:,32]

# Ignore flow velocity, set to zero vector.
flow_velocity = lambda x, y, z: Vector3D(0,0,0)

# Set Ti = Te
ion_temperature = client.get("AYC_TE", PULSE)
radius_data = client.get("ayc_r", PULSE) #spatial extent of the data
ion_time_index = equilibrium._find_nearest(ion_temperature.dims[0].data, TIME) #find the time index we're interested in

#get the time slice
ion_temp = ion_temperature.data[ion_time_index,:]
radius = radius_data.data[ion_time_index,:]

#Interpolate psi onto the spatial extent of data
interp_psi = Interpolate1DCubic(r, psi_z0)
new_psi = np.zeros((len(radius)))

#find the values of psi that match to radial data points
for i in range(len(radius)):
    new_psi[i] = interp_psi(radius[i])

#find where the plasma is, then take the indices from the center to the edge
mask = np.where(new_psi<=1.0)[0][:]
center_to_edge = int(len(mask)/2)
mask = mask[center_to_edge:]

ion_temperature_data = ion_temperature.data[ion_time_index,:]
ion_temperature_data[np.isnan(ion_temperature_data)]=0.
print("Ti between {} and {} eV".format(ion_temperature_data.min(), ion_temperature_data.max()))

ion_temperature_data = ion_temperature_data[mask] #get ion temperature from the center to the edge of the plasma

plt.figure()
plt.plot(new_psi[mask], ion_temperature_data)
plt.title('ion temp')


ion_temperature_psi = Interpolate1DCubic(new_psi[mask], ion_temperature_data, extrapolate=True)
ion_temperature = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, ion_temperature_psi), inside_lcfs))


# Plot the ion temperature

xl, xu = (0.0,2.0)
yl, yu = (-2.0, 2.0)

ion_temp = np.zeros((500,500))
xrange = np.linspace(xl, xu, 500)
yrange = np.linspace(yl, yu, 500)

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        ion_temp[j,i] = ion_temperature(x, 0.0, y)

plt.figure()
plt.imshow(ion_temp)
plt.title('Ion temperature (eV) in poloidal plane')
plt.colorbar()

### ELECTRON DENSITY ####

# Set Ti = Te
electron_density_data = client.get("AYC_NE", PULSE)
electron_time_index = equilibrium._find_nearest(electron_density_data.dims[0].data, TIME)

#get the time slice
electron_density = electron_density_data.data[electron_time_index,:]

electron_density[np.isnan(electron_density)]=0. #get rid of nans in array

print("Ne between {} and {} eV".format(electron_density.min(), electron_density.max()))

psi_insidelcfs = new_psi[mask]
psi_insidelcfs = np.append(psi_insidelcfs, [1.0])

ne_lcfs = electron_density[mask]
ne_lcfs = np.append(ne_lcfs, [0.0])

plt.figure()
plt.plot(psi_insidelcfs, ne_lcfs)
plt.title('electron density')

e_density_psi = Interpolate1DCubic(psi_insidelcfs, ne_lcfs, extrapolate=True)

e_density = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, e_density_psi), inside_lcfs))


# Plot electron density
ne = np.zeros((500,500))

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        ne[j,i] = e_density(x, 0.0, y)

plt.figure()
plt.imshow(ne)
plt.colorbar()
plt.title('Electron density in poloidal plane')


# set to 1% electron density
density_c6_data = 0.01*ne_lcfs

plt.figure()
plt.plot(psi_insidelcfs, density_c6_data)
plt.title('c6 density')

density_c6_psi = Interpolate1DCubic(psi_insidelcfs, density_c6_data, extrapolate=True)
density_c6 = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, density_c6_psi), inside_lcfs))
density_d = lambda x, y, z: e_density(x, y, z) - 6 * density_c6(x, y, z)

# Plot carbon density
c6 = np.zeros((500,500))
d = np.zeros((500,500))

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        d[j,i] = density_d(x, 0.0, y)
        c6[j,i] = density_c6(x, 0.0, y)

plt.figure()
plt.imshow(c6)
plt.title('c6 density')

plt.figure()
plt.imshow(d)
plt.colorbar()
plt.title('deuterium density in poloidal plane')

#define distributions
d_distribution = Maxwellian(density_d, ion_temperature, flow_velocity, deuterium.atomic_weight * atomic_mass)
c6_distribution = Maxwellian(density_c6, ion_temperature, flow_velocity, carbon.atomic_weight * atomic_mass)
e_distribution = Maxwellian(electron_density, ion_temperature, flow_velocity, electron_mass)

d1_species = Species(elements.deuterium, 1, d_distribution)
c6_species = Species(carbon, 6, c6_distribution)

#define plasma parameters - electron distribution, impurity composition and B field from EFIT
plasma.electron_distribution = e_distribution
plasma.composition = [d1_species, c6_species]
plasma.b_field = VectorAxisymmetricMapper(equil_time_slice.b_field)
sigma = 0.25
integration_step = 0.02

#define the plasma geometry

plasma.integrator.step = integration_step
plasma.integrator.min_samples = 1000
plasma.atomic_data = adas
plasma.geometry = Cylinder(sigma * 2, sigma * 10.0)
plasma.geometry_transform = translate(0, -sigma * 5.0, 0) * rotate(0, 90, 0)

# # # ########################### NBI CONFIGURATION ############################# #

#Geometry
south_pos = Point3D(0.188819939,-6.88824321,0.0)    #Position of PINI grid center
duct_pos = Point3D(0.539, -1.926, 0.00)         #position of beam duct
south_pos.vector_to(duct_pos)                   #beam vector
beam_axis = south_pos.vector_to(duct_pos).normalise()

up = Vector3D(0, 0, 1)
beam_rotation = rotate_basis(beam_axis, up)

beam_position = translate(south_pos.x, south_pos.y, south_pos.z)

beam_full = Beam(parent=world,transform=beam_position*beam_rotation)
beam_full.plasma = plasma
beam_full.atomic_data = adas
beam_full.energy = 65000
beam_full.power = 3e6
beam_full.element = elements.deuterium
beam_full.sigma = 0.025
beam_full.divergence_x = 0  #0.5
beam_full.divergence_y = 0  #0.5
beam_full.length = 10.0
beam_full.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam_full.models = [
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
]
beam_full.integrator.step = integration_step
beam_full.integrator.min_samples = 10

beam_half = Beam(parent=world, transform=beam_position*beam_rotation)
beam_half.plasma = plasma
beam_half.atomic_data = adas
beam_half.energy = 65000 / 2.
beam_half.power = 3e6
beam_half.element = elements.deuterium
beam_half.sigma = 0.025
beam_half.divergence_x = 0  #0.5
beam_half.divergence_y = 0  #0.5
beam_half.length = 10.0
beam_half.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam_half.models = [
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
]
beam_half.integrator.step = integration_step
beam_half.integrator.min_samples = 10

beam_third = Beam(parent=world, transform=beam_position*beam_rotation)
beam_third.plasma = plasma
beam_third.atomic_data = adas
beam_third.energy = 65000 / 3.
beam_third.power = 3e6
beam_third.element = elements.deuterium
beam_third.sigma = 0.025
beam_third.divergence_x = 0  #0.5
beam_third.divergence_y = 0  #0.5
beam_third.length = 10.0
beam_third.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam_third.models = [
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
]
beam_third.integrator.step = integration_step
beam_third.integrator.min_samples = 10


to_local = (beam_position*beam_rotation).inverse()

length = []
full_energy = []
half_energy = []
third_energy = []

d_gas_density = []
c_gas_density = []
e_gas_density = []

xc = []
yc = []
zc = []

t_span = np.linspace(0, 2, 500)

for t in t_span:
    sample_point = south_pos + beam_axis * t
    local_point = sample_point.transform(to_local)
    try:

        d_gas_density.append(density_d(sample_point.x, sample_point.y, sample_point.z))
        c_gas_density.append(density_c6(sample_point.x, sample_point.y, sample_point.z))
        e_gas_density.append(e_density(sample_point.x, sample_point.y, sample_point.z))

        full_energy.append(beam_full.density(local_point.x, local_point.y, local_point.z))
        half_energy.append(beam_half.density(local_point.x, local_point.y, local_point.z))
        third_energy.append(beam_third.density(local_point.x, local_point.y, local_point.z))

        xc.append(sample_point.x)
        yc.append(sample_point.y)
        zc.append(sample_point.z)
        length.append(t)

    except ValueError:
        continue

beam = {'xc': xc, 'yc': yc, 'zc': zc,
        'full_beam_density': full_energy, 'half_beam_density': half_energy, 'third_beam_density': third_energy}

los = Point3D(-1.742, 1.564, 0.179) #camera position
direction = Vector3D(0.919, -0.389, -0.057).normalise() #optical axis
los = los + direction * 0.9
up = Vector3D(1, 1, 1)
pos = translate(los.x, los.y, los.z) * rotate_basis(direction, up)

print(los)

view = {'view':los, 'direction':direction, 'pos':pos}

# Uncomment to write the results to file.
# with open('/home/sgibson/Desktop/beam_sample2.pkl', 'wb') as f:
#     pickle.dump(beam, f, pickle.HIGHEST_PROTOCOL)

# with open('/home/sgibson/Desktop/view.pkl', 'wb') as f:
#     pickle.dump(view, f, pickle.HIGHEST_PROTOCOL)

# idl.put('xc',xc)
# idl.put('yc',yc)
# idl.put('zc',zc)
# idl.put('full_energy',full_energy)
# idl.put('half_energy',half_energy)
# idl.put('third_energy',third_energy)

# idl.execute('save,xc,yc,zc,full_energy,half_energy,third_energy,filename="/home/sgibson/Project/msesim/beam_params.xdr"')

# ############################### OBSERVATION ###############################
# print('Observation')

duct_pos = Point3D(0.539, -1.926, 0) #Position of beam
south_pos.vector_to(duct_pos)
beam_axis = south_pos.vector_to(duct_pos).normalise()

los = Point3D(-1.742, 1.564, 0.179) #camera position
direction = Vector3D(0.919, -0.389, -0.057).normalise() #optical axis
los = los + direction * 0.9
up = Vector3D(1, 1, 1)
translate(los.x, los.y, los.z) * rotate_basis(direction, up)

camera = PinholeCamera((256, 256), fov=52.9, parent=world, transform=translate(0, 0, 3)*rotate_basis(Vector3D(0, 0, -1), Vector3D(1, 0, 0)) )
camera.pixel_samples = 50
camera.spectral_bins = 15
camera.render_engine = SerialEngine()
camera.observe()

plt.figure()
plt.plot(length, d_gas_density, label='d')
plt.plot(length, c_gas_density, label='c')
plt.plot(length, e_gas_density, label='e')
plt.legend()
plt.xlabel('Distance along beam line (m)')
plt.ylabel('density')

plt.figure()
plt.plot(length, full_energy, label='full energy')
plt.plot(length, half_energy, label='half energy')
plt.plot(length, third_energy, label='third energy')
plt.xlabel('Distance along beam line (m)')
plt.ylabel('Beam component density')
plt.legend()
plt.title('Attenuation of beam components')
plt.ioff()
plt.show()
