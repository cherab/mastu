"""
This script demonstrates how to calculate the volumetric sensitivity
matrix for the bolometer foils. For low resolution grids (~1k cells,
~10k rays), the calculation is fast enough to be performed in an
interactive session.  For higher resolution grids (~1M cells) or higher
ray counts (~1M rays) this should be submitted as a batch job
instead. See the `core_highres.sh` script for an example.

Note that the noise in the sensitivity matrix varies inversely with both
the size of the grid cells and the number of rays. So smaller cells will
require higher ray counts (to get sufficient rays to pass through each
cell).
"""
import os
import sys
import multiprocessing
import numpy as np

from raysect.optical import World
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.primitive import Mesh, import_obj, Cylinder
from raysect.core import translate

from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D

from cherab.mastu.machine.mast_m9_cad_files import MAST_FULL_MESH
from cherab.mastu.machine.mastu_cad_files import MASTU_FULL_MESH
from cherab.mastu.bolometry import load_default_bolometer_config, load_standard_voxel_grid

# If running as a batch job, use only the requested number of processes
NSLOTS = os.environ.get('NSLOTS', multiprocessing.cpu_count())


def load_vessel_world(mesh_parts, shift_p5=False):
    """Load the world containing the vessel mesh parts.

    <mesh_parts> is a list of filenames containing mesh files in either
    RSM or OBJ format, which are to be loaded into the world.

    If shift_p5 is True, the mesh files representing the P5 coils will
    have a downward shift applied to them to account for the UEP sag.
    This is described in CD/MU/04783.

    Returns world, the root of the scenegraph.
    """
    world = World()
    for path, _ in mesh_parts:
        print("importing {}  ...".format(os.path.split(path)[1]))
        filename = os.path.split(path)[-1]
        name, ext = filename.split('.')
        if 'P5_' in path and shift_p5:
            p5_zshift = -0.00485  # From CD/MU/04783
            transform = translate(0, 0, p5_zshift)
        else:
            transform = None
        if ext.lower() == 'rsm':
            Mesh.from_file(path, parent=world, material=AbsorbingSurface(),
                           transform=transform, name=name)
        elif ext.lower() == 'obj':
            import_obj(path, parent=world, material=AbsorbingSurface(), name=name)
        else:
            raise ValueError("Only RSM and OBJ meshes are supported.")
    # Add a solid cylinder at R=0 to prevent rays finding their way through the
    # gaps in the centre column armour. This is only necessary for the MAST
    # meshes, but it does no harm to add it to MAST-U meshes too
    height = 6
    radius = 0.1
    Cylinder(radius=radius, height=height, parent=world,
             transform=translate(0, 0, -height / 2),
             material=AbsorbingSurface(), name='Blocking Cylinder')
    return world


def calculate_and_save_sensitivities(grid_name, camera, mesh_parts,
                                     shift_p5=False, camera_transform=None):
    """
    Calculate the sensitivity matrices for an entire bolometer camera,
    and save to a Numpy save file.

    The output file is named as follows:
    <grid>_<camera>_bolo.npy

    Parameters:
    grid: the name of the reconstruction grid to load
    camera: the name of the bolometer camera
    mesh_parts: the list of mesh files to read and load into the world
    """
    world = load_vessel_world(mesh_parts, shift_p5)
    # Trim off metadata from camera if it exists. This metadata is only for the
    # name of the save file.
    cam_nometa = camera.split("-")[0]
    if grid_name == "sxdl":
        bolo_name = "SXDL - {}".format(cam_nometa)
    elif grid_name in("core", "core_high_res"):
        bolo_name = "CORE - {}".format(cam_nometa)
    else:
        raise ValueError("Only 'sxd', 'core' and 'core_high_res' grids supported.")
    # grid = load_standard_voxel_grid(grid_name, parent=world)
    file_name = "rtm_{}_{}_bolo.npy".format(
        grid_name.replace("_", "-"), camera.lower()
    )
    bolo = load_default_bolometer_config(bolo_name, parent=world, shot=50000)

    if camera_transform is not None:
        bolo.transform = camera_transform * bolo.transform

    if grid_name == 'sxdl':
        rtc = RayTransferCylinder(radius_inner=0.5, radius_outer=2, height=0.6,
                                  transform=translate(0, 0, -2.1),
                                  mask=None, n_radius=150, n_height=60,
                                  parent=world)
    elif grid_name == 'core':
        rtc = RayTransferCylinder(radius_inner=0.25, radius_outer=1.5431, height=3.1,
                                  transform=translate(0, 0, -1.549),
                                  mask=None, n_radius=30, n_height=70,
                                  parent=world)
    elif grid_name == 'core_high_res':
        rtc = RayTransferCylinder(radius_inner=0.25, radius_outer=1.5431, height=3.1,
                                  transform=translate(0, 0, -1.549),
                                  mask=None, n_radius=750, n_height=1550,
                                  parent=world)
    else:
        raise ValueError("Invalid grid name {}".format(grid_name))
    sensitivities = np.zeros((len(bolo), rtc.bins))
    for i, detector in enumerate(bolo):
        detector.render_engine.processes = NSLOTS
        print('calculating detector {}'.format(detector.name))
        detector.pixel_samples = 10000
        detector.spectral_bins = rtc.bins
        detector.min_wavelength = 400
        detector.max_wavelength = detector.min_wavelength + 1
        detector.pipelines = [RayTransferPipeline0D()]
        detector.observe()
        sensitivities[i] = detector.pipelines[0].matrix
    np.save(file_name, sensitivities)


def main():
    camera = sys.argv[1]
    shift_p5 = False
    camera_transform = None

    if camera in ("Outer", "Upper"):
        MESH_PARTS = MASTU_FULL_MESH
        grid = "sxdl"
    elif camera in ("Poloidal", "Tangential"):
        MESH_PARTS = MASTU_FULL_MESH
        shift_p5 = True
        grid = "core"
    elif camera in ("PoloidalHighRes", "TangentialHighRes"):
        MESH_PARTS = MASTU_FULL_MESH
        shift_p5 = True
        grid = "core_high_res"
        # Trim off the HighRes suffix from the camera
        camera = camera[:-7]
    elif camera == "Poloidal-MAST":
        MESH_PARTS = MAST_FULL_MESH
        # MAST sensors were displaced by 100mm in the y direction compared with
        # the main chamber sensors at the beginning of MAST-U operation
        camera_transform = translate(0, 0.1, 0)
        grid = "core"
    else:
        raise ValueError("The following cameras are supported: "
                         "'Outer', 'Upper', 'Poloidal', 'Tangential', "
                         "'PoloidalHighRes', 'TangentialHighRes', "
                         "'Poloidal-MAST'")

    calculate_and_save_sensitivities(
        grid, camera, MESH_PARTS, shift_p5, camera_transform
    )


if __name__ == "__main__":
    main()
