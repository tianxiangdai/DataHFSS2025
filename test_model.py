import numpy as np
from matplotlib import pyplot as plt

from mylibs.visualization import Plotter
from mylibs.models import S1T4ForceCrossCW, ModelParameter

if __name__ == "__main__":

    r_OC = np.array([0, -0.35, 0.2], float)
    # r_OC = np.array([0, -0.35, 0.15], float)
    r_OF = np.array([0, 0, 0.06], float)  # camera focal point
    e_x_cam = np.array([1, 0, 0], float)
    e_z_cam = r_OF - r_OC
    e_z_cam /= np.linalg.norm(e_z_cam)
    e_y_cam = np.cross(e_z_cam, e_x_cam)
    zoom = 1
    # zoom = 1.5
    fx = fy = 2635.5177
    px, py = 3840, 2160  # camera 4k resolution
    window_size = (960, 540)
    # window_size = (540, 540)
    cam_view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)
    
    model = S1T4ForceCrossCW()
    plotter = Plotter(
        model.system,
        window_size,
    )
    cam = plotter.camera
    cam.SetViewAngle(cam_view_angle)
    cam.SetPosition(*r_OC)
    cam.SetFocalPoint(*r_OF)
    cam.SetViewUp(*(-e_y_cam))
    cam.SetClippingRange(0.01, 1)
    cam.Zoom(zoom)
    plotter.show()

    r_OP = model.apply_forces(
        np.array([50, 0, 0, 0]),
        eval_keys=["r_OP"],
        force_steps=100,
        verbose=True,
        ret_all_steps=True,
    )
