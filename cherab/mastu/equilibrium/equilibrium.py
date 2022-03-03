
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

"""
MAST-U equilibrium data reading routines
"""

import pyuda
import numpy as np

from raysect.core import Point2D
from cherab.tools.equilibrium import EFITEquilibrium


class MASTEquilibrium:
    """
    Reads MAST EFIT equilibrium data and provides object access to each timeslice.

    :param pulse: MAST pulse number.
    """

    def __init__(self, pulse):

        if pulse >= 40000:
            raise ValueError("MASTEquilibrium on supports pulses before 40000. "
                             "Use MASTUEquilibrium instead for pulses after 40000.")

        self.client = pyuda.Client()  # get the pyuda client

        # Poloidal magnetic flux per toroidal radian as a function of (Z,R) and timebase
        self.psi = self.client.get("efm_psi(r,z)", pulse)

        self.time_slices = self.psi.dims[0].data

        # Psi grid axes f(nr), f(nz)
        self.r = self.client.get("efm_grid(r)", pulse)
        self.z = self.client.get("efm_grid(z)", pulse)

        # f profile
        self.f = self.client.get("efm_f(psi)_(c)", pulse)# Poloidal current flux function, f=R*Bphi; f(psin, C)

        # q profile
        self.q = self.client.get("efm_q(psi)_(c)", pulse)

        self.psi_r = self.client.get("efm_psi(r)", pulse) #poloidal magnetic flux per toroidal radian as a function of radius at Z=0

        # Poloidal magnetic flux per toroidal radian at the plasma boundary and magnetic axis
        self.psi_lcfs = self.client.get("efm_psi_boundary", pulse)
        self.psi_axis = self.client.get("efm_psi_axis", pulse)

        # Plasma current
        self.plasma_current = self.client.get("efm_plasma_curr(C)", pulse)

        # Reference vaccuum toroidal B field at R = efm_bvac_r
        self.b_vacuum_magnitude = self.client.get("efm_bvac_val", pulse)

        self.b_vacuum_radius = self.client.get("efm_bvac_r", pulse)

        # Magnetic axis co-ordinates
        self.axis_coord_r = self.client.get("efm_magnetic_axis_r", pulse)
        self.axis_coord_z = self.client.get("efm_magnetic_axis_z", pulse)

        # X point coordinates
        xpoint1r = self.client.get("efm_xpoint1_r(c)", pulse).data
        xpoint1z = self.client.get("efm_xpoint1_z(c)", pulse).data
        xpoint2r = self.client.get("efm_xpoint2_r(c)", pulse).data
        xpoint2z = self.client.get("efm_xpoint2_z(c)", pulse).data
        self.xpoints = [
            (Point2D(r1, z1), Point2D(r2, z2))
            for (r1, z1, r2, z2) in zip(xpoint1r, xpoint1z, xpoint2r, xpoint2z)
        ]

        #minor radius
        self.minor_radius = self.client.get("efm_minor_radius", pulse)

        #lcfs boundary polygon
        self.lcfs_poly_r = self.client.get("efm_lcfs(r)_(c)", pulse)
        self.lcfs_poly_z = self.client.get("efm_lcfs(z)_(c)", pulse)

        # Number of LCFS co-ordinates
        self.nlcfs = self.client.get("efm_lcfs(n)_(c)", pulse)

        # limiter boundary polygon
        self.limiter_poly_r = self.client.get("efm_limiter(r)", pulse)
        self.limiter_poly_z = self.client.get("efm_limiter(z)", pulse)

        # Number of limiter co-ordinates
        self.nlimiter = self.client.get("efm_limiter(n)", pulse)

        # time slices when plasma is present
        self.plasma_times = self.client.get("efm_ip_times", pulse)

        self.time_range = self.time_slices.min(), self.time_slices.max()

    def time(self, time):
        """
        Returns an equilibrium object for the time-slice closest to the requested time.

        The specific time-slice returned is held in the time attribute of the returned object.

        :param time: The equilibrium time point.
        :returns: An EFIT Equilibrium object.
        """

        # locate the nearest time point and fail early if we are outside the time range of the data

        try:
            index = self._find_nearest(self.time_slices, time)
            # Find the index in the time array defined as when the plasma is present
            plasma_index = self._find_nearest(self.plasma_times.data, time)
        except IndexError:
            raise ValueError('Requested time lies outside the range of the data: [{}, {}]s.'.format(*self.time_range))

        b_vacuum_radius = self.b_vacuum_radius.data[index]

        time = self.time_slices[index]

        psi = np.transpose(self.psi.data[index,:,:]) #transpose psi to get psi(R,Z) instead of psi(Z,R)

        psi_lcfs = self.psi_lcfs.data[plasma_index]

        psi_axis = self.psi_axis.data[plasma_index]

        print('psi_axis', psi_axis)

        f_profile_psin = self.f.dims[1].data
        self.f_profile_psin = f_profile_psin
        f_profile_magnitude = self.f.data[plasma_index, :]
        f_profile = np.asarray([f_profile_psin, f_profile_magnitude])

        q_profile_magnitude = self.q.data[plasma_index]
        q_profile_psin = self.q.dims[1].data
        q_profile = np.asarray([q_profile_psin, q_profile_magnitude])

        axis_coord = Point2D(self.axis_coord_r.data[plasma_index], self.axis_coord_z.data[plasma_index])

        b_vacuum_magnitude = self.b_vacuum_magnitude.data[index]

        lcfs_poly_r = self.lcfs_poly_r.data[plasma_index,:]
        lcfs_poly_z = self.lcfs_poly_z.data[plasma_index,:]

        # Get the actual co-ordinates of the LCFS
        lcfs_points = self.nlcfs.data[plasma_index]

        #Filter out padding in the LCFS coordinate arrays
        lcfs_poly_r = lcfs_poly_r[0:lcfs_points]
        lcfs_poly_z = lcfs_poly_z[0:lcfs_points]

        # convert raw lcfs poly coordinates into a polygon object
        lcfs_polygon = self._process_efit_lcfs_polygon(lcfs_poly_r, lcfs_poly_z)
        lcfs_polygon = np.ascontiguousarray(lcfs_polygon.T)  # 2xN contiguous
        self.lcfs_polygon = lcfs_polygon

        limiter_poly_r = self.limiter_poly_r.data.squeeze()
        limiter_poly_z = self.limiter_poly_z.data.squeeze()

        # Get the actual co-ordinates of the limiter
        limiter_points = self.nlimiter.data.item()

        #Filter out padding in the LIMITER coordinate arrays
        limiter_poly_r = limiter_poly_r[0:limiter_points]
        limiter_poly_z = limiter_poly_z[0:limiter_points]

        # convert raw limiter poly coordinates into a polygon object
        limiter_polygon = self._process_efit_lcfs_polygon(limiter_poly_r, limiter_poly_z)
        limiter_polygon = np.ascontiguousarray(limiter_polygon.T)  # 2xN contiguous
        self.limiter_polygon = limiter_polygon

        r = self.r.data.squeeze()
        z = self.z.data.squeeze()

        xpoints = self.xpoints[plasma_index]

        # MAST-U EFIT has no reliably-available strike point data (ASF not always available)
        strike_points = []

        minor_radius = self.minor_radius.data[plasma_index]

        print('minor radius', minor_radius)

        return EFITEquilibrium(r, z, psi, psi_axis, psi_lcfs, axis_coord,
                               xpoints, strike_points, f_profile, q_profile,
                               b_vacuum_radius, b_vacuum_magnitude,
                               lcfs_polygon, limiter_polygon, time)

    @staticmethod
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data.")

        index = np.searchsorted(array, value, side="left")

        if (value - array[index])**2 < (value - array[index + 1])**2:
            return index
        else:
            return index + 1

    @staticmethod
    def _process_efit_lcfs_polygon(poly_r, poly_z):

        if poly_r.shape != poly_z.shape:
            raise ValueError("EFIT LCFS polygon coordinate arrays are inconsistent in length.")

        n = poly_r.shape[0]
        if n < 2:
            raise ValueError("EFIT LCFS polygon coordinate contain less than 2 points.")

        # boundary polygon contains redundant points that must be removed
        unique = (poly_r != poly_r[0]) | (poly_z != poly_z[0])
        unique[0] = True  # first point must be included!
        poly_r = poly_r[unique]
        poly_z = poly_z[unique]

        # generate single array containing coordinates
        poly_coords = np.zeros((len(poly_r), 2))
        poly_coords[:, 0] = poly_r
        poly_coords[:, 1] = poly_z

        # magic number for vocel_coef from old code
        return poly_coords


class MASTUEquilibrium:
    """
    Reads MAST-U EFIT equilibrium data and provides object access to each timeslice.

    :param pulse: MAST pulse number.
    """

    def __init__(self, pulse):

        if isinstance(pulse, int) and pulse < 40000:
            raise ValueError("MAST-U Equilibria only supported for pulse 40000 onwards.")

        self.client = pyuda.Client()  # get the pyuda client

        # Poloidal magnetic flux per toroidal radian as a function of (time, R, Z)
        self.psi = self.client.get("/epm/output/profiles2D/poloidalFlux", pulse)

        self.time_slices = self.psi.dims[0].data
        self.time_range = self.time_slices.min(), self.time_slices.max()

        # Psi grid axes f(nr), f(nz)
        self.r = self.client.get("/epm/output/profiles2D/r", pulse)
        self.z = self.client.get("/epm/output/profiles2D/z", pulse)

        # f profile: poloidal current flux function, f=R*Bphi; f(psin, C)
        self.f = self.client.get("/epm/output/fluxFunctionProfiles/ffPrime", pulse)

        # q profile
        self.q = self.client.get("/epm/output/fluxFunctionProfiles/q", pulse)

        # Poloidal magnetic flux per toroidal radian as a function of radius at Z=0
        self.psi_r = self.client.get("/epm/output/fluxFunctionProfiles/poloidalFlux", pulse)

        # Poloidal magnetic flux per toroidal radian at the plasma boundary and magnetic axis
        self.psi_lcfs = self.client.get("/epm/output/globalParameters/psiBoundary", pulse)
        self.psi_axis = self.client.get("/epm/output/globalParameters/psiAxis", pulse)

        # Plasma current
        self.plasma_current = self.client.get("/epm/output/globalParameters/plasmaCurrent", pulse)

        # Reference vaccuum toroidal B field at R = efm_bvac_r
        self.b_vacuum_magnitude = self.client.get("/epm/output/globalParameters/bvacRmag", pulse)

        # Magnetic axis co-ordinates
        self.axis_coord_r = self.client.get("/epm/output/globalParameters/magneticAxis/R", pulse)
        self.axis_coord_z = self.client.get("/epm/output/globalParameters/magneticAxis/Z", pulse)

        # Reference vacuum field is at the magnetic axis.
        self.b_vacuum_radius = self.axis_coord_r

        # X point and strike point coordinates.
        nxp = self.client.get("/epm/output/separatrixGeometry/xpointCount", pulse).data
        rxp = self.client.get("/epm/output/separatrixGeometry/xpointR", pulse).data
        zxp = self.client.get("/epm/output/separatrixGeometry/xpointZ", pulse).data
        rsp = self.client.get("/epm/output/separatrixGeometry/strikepointR", pulse).data
        zsp = self.client.get("/epm/output/separatrixGeometry/strikepointZ", pulse).data
        self.xpoints = []
        self.strike_points = []
        for n, rx, zx, rs, zs in zip(nxp, rxp, zxp, rsp, zsp):
            if np.isnan(n):
                n = 0
            n = int(n)
            self.xpoints.append([Point2D(x, y) for (x, y) in zip(rx[:n], zx[:n])])
            # Each X point will have 2 associated strike points.
            self.strike_points.append([Point2D(x, y) for (x, y) in zip(rs[:2*n], zs[:2*n])])

        #minor radius
        self.minor_radius = self.client.get("/epm/output/separatrixGeometry/minorRadius", pulse)

        #lcfs boundary polygon
        self.lcfs_poly_r = self.client.get("/epm/output/separatrixGeometry/rBoundary", pulse)
        self.lcfs_poly_z = self.client.get("/epm/output/separatrixGeometry/zBoundary", pulse)

        # limiter boundary polygon
        self.limiter_poly_r = self.client.get("/epm/input/limiter/rValues", pulse)
        self.limiter_poly_z = self.client.get("/epm/input/limiter/zValues", pulse)

    @staticmethod
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data.")

        index = np.searchsorted(array, value, side="left")

        if (value - array[index])**2 < (value - array[index + 1])**2:
            return index
        else:
            return index + 1

    def time(self, time):
        """
        Returns an equilibrium object for the time-slice closest to the requested time.

        The specific time-slice returned is held in the time attribute of the returned object.

        :param time: The equilibrium time point.
        :returns: An EFIT Equilibrium object.
        """

        # locate the nearest time point and fail early if we are outside the time range of the data
        tmin = self.time_slices.min()
        tmax = self.time_slices.max()
        if not tmin <= time <= tmax:
            raise ValueError('Requested time lies outside the range of the data: [{}, {}]s.'.format(tmin, tmax))
        index = np.argmin(abs(self.time_slices - time))

        time = self.time_slices[index]

        psi = self.psi.data[index]
        psi_lcfs = self.psi_lcfs.data[index]
        psi_axis = self.psi_axis.data[index]

        f_profile_psin = self.f.dims[1].data
        f_profile_magnitude = self.f.data[index, :]
        f_profile = np.asarray([f_profile_psin, f_profile_magnitude])

        q_profile_magnitude = self.q.data[index]
        q_profile_psin = self.q.dims[1].data
        q_profile = np.asarray([q_profile_psin, q_profile_magnitude])

        axis_coord = Point2D(self.axis_coord_r.data[index], self.axis_coord_z.data[index])

        b_vacuum_magnitude = self.b_vacuum_magnitude.data[index]
        b_vacuum_radius = self.b_vacuum_radius.data[index]

        lcfs_poly_r = self.lcfs_poly_r.data[index]
        lcfs_poly_z = self.lcfs_poly_z.data[index]
        # LCFS always has the same number of points.
        lcfs_polygon = np.stack((lcfs_poly_r, lcfs_poly_z), axis=0)
        if np.all(lcfs_polygon[:, 0] == lcfs_polygon[:, -1]):
            # First and last points are the same: need an open polygon for Cherab.
            lcfs_polygon = lcfs_polygon[:, :-1]

        limiter_poly_r = self.limiter_poly_r.data
        limiter_poly_z = self.limiter_poly_z.data
        limiter_polygon = np.stack((limiter_poly_r, limiter_poly_z), axis=0)
        if np.all(limiter_polygon[:, 0] == limiter_polygon[:, -1]):
            # First and last points are the same: need an open polygon for Cherab.
            limiter_polygon = limiter_polygon[:, :-1]

        r = self.r.data
        z = self.z.data

        xpoints = self.xpoints[index]
        strike_points = self.strike_points[index]

        return EFITEquilibrium(r, z, psi, psi_axis, psi_lcfs, axis_coord,
                               xpoints, strike_points, f_profile, q_profile,
                               b_vacuum_radius, b_vacuum_magnitude,
                               lcfs_polygon, limiter_polygon, time)
