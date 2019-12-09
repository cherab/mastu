
import pyuda

from raysect.core import Point3D, Vector3D
from raysect.primitive import Mesh
from cherab.tools.observers.bolometry import BolometerCamera, BolometerFoil, BolometerSlit
from cherab.mastu.machine import CORE_BOLOMETERS, SXD_BOLOMETERS


def load_default_bolometer_config(bolometer_id, parent=None, shot=50000, override_material=None):
    """
    Load the bolometer confituration for a given bolometer ID.

    :param bolometer_id: the name of the bolometer camera,
    in the form "chamber - camera"
    :param parent: the parent node of the resulting bolometer node
    :param shot: read the bolometer state at the given shot
    <shot> can be either a shot number, in which case the geometry is
    looked up in pyuda, or a string containing the path to a local
    netCDF file, e.g. '/home/jlovell/Bolometer/bolo_geom_fromscript.nc',
    in which case the geometry is loaded from that file. The file must
    conform to the same specification as that in the machine
    description, described in CD/MU/05410.
    :param override_material: a Raysect material to assign to the camera
    mesh geometry, instead of the default metal.

    :return bolometer: the BolometerCamera object for the requested camera.
    """
    try:
        chamber, camera_name = [elem.strip() for elem in bolometer_id.split('-')]
    except KeyError:
        raise ValueError("Bolometer camera ID '{}' not recognised.".format(bolometer_id))

    if chamber.lower() not in ('sxdl', 'core'):
        raise ValueError('Chamber must be "SXDL" or "CORE"')

    client = pyuda.Client()
    try:
        # Need to pass no_cal=True to UDA client, since none of the Cherab
        # meshes have shifts applied to them yet.
        bolometers = client.geometry('/bolo/{}'.format(chamber.lower()), shot,
                                     cartesian_coords=True, no_cal=True).data
    except AttributeError:
        raise pyuda.UDAException("Couldn't retrieve bolometer geometry data from UDA.")
    if chamber.lower() == 'sxdl':
        if camera_name.lower() == 'outer':
            mesh_geometry_data = SXD_BOLOMETERS[0]
        elif camera_name.lower() == 'upper':
            mesh_geometry_data = SXD_BOLOMETERS[1]
        else:
            raise ValueError('For SXDL, camera name must be "Outer" or "Upper"')
    elif chamber.lower() == 'core':
        if camera_name.lower() == 'poloidal':
            mesh_geometry_data = CORE_BOLOMETERS[0]
        elif camera_name.lower() == 'tangential':
            mesh_geometry_data = CORE_BOLOMETERS[1]
        else:
            raise ValueError('For Core, camera name must be "Poloidal" or "Tangential"')

    camera_material = override_material or mesh_geometry_data[1]
    camera_geometry = Mesh.from_file(mesh_geometry_data[0], material=camera_material)

    camera_id = 'MAST-U {} - {} Bolometer'.format(chamber, camera_name)
    bolometer_camera = BolometerCamera(name=camera_id, parent=parent,
                                       camera_geometry=camera_geometry)
    # FIXME: When reading from local files with UDA, need an extra data child.
    # This is due to a bug in the pyuda geometry wrapper for local files.
    try:
        slits = bolometers['/slits/data']
    except KeyError:
        slits = bolometers['/data/slits/data']
    slit_objects = {}
    for slit in slits:
        try:
            curvature_radius = slit.curvature_radius
        except AttributeError:
            curvature_radius = 0
        # Only include slits in the requested camera
        # Slit IDs are of the form "MAST-U <chamber> - <camera> Slit <N>"
        slit_camera = slit.id.split('-')[2].split()[0]
        if slit_camera != camera_name:
            continue
        slit_objects[slit.id] = BolometerSlit(
            basis_x=uda_to_vector(slit['basis_1']),
            basis_y=uda_to_vector(slit['basis_2']),
            centre_point=uda_to_point(slit['centre_point']),
            dx=slit.width,
            dy=slit.height,
            curvature_radius=curvature_radius,
            slit_id=slit.id,
            parent=bolometer_camera,
        )

    try:
        foils = bolometers['/foils/data']
    except KeyError:
        foils = bolometers['/data/foils/data']
    for foil in foils:
        # Only include foils in the requested camera
        # Foil IDs are of the form "MAST-U <chamber> - <camera> CH<N>"
        foil_camera = foil.id.split('-')[2].split()[0]
        if foil_camera != camera_name:
            continue
        foil = BolometerFoil(
            basis_x=uda_to_vector(foil['basis_1']),
            basis_y=uda_to_vector(foil['basis_2']),
            centre_point=uda_to_point(foil['centre_point']),
            dx=foil.width,
            dy=foil.height,
            curvature_radius=foil.curvature_radius,
            detector_id=foil.id,
            slit=slit_objects[foil.slit_id],
            parent=bolometer_camera,
        )
        bolometer_camera.add_foil_detector(foil)

    return bolometer_camera


def uda_to_point(struct):
    """Convert the x, y and z components of a UDA structure into a Point3D."""
    return Point3D(struct.x, struct.y, struct.z)


def uda_to_vector(struct):
    """Convert the x, y and z components of a UDA structure into a Vector3D."""
    return Vector3D(struct.x, struct.y, struct.z)
