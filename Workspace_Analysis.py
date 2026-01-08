import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
from xarray import load_dataset
import trimesh


data_dir = "Experimental_data/Workspace"

mm2pt = 72 / 25.4  # mm to points


def cyl_coord(r_OP):
    rho = np.linalg.norm(r_OP[:, :2], axis=1) + 1e-15
    z = r_OP[:, 2]
    phi = np.arccos(r_OP[:, 0] / rho)
    phi[(r_OP[:, 1] < 0)] *= -1
    phi[phi < -1e-3] += np.pi * 2
    phi[rho < 1e-6] = 0
    return rho, z, phi


###########
# load data
###########
la_min = 0
la_max = 50
step = 11
# raw data
file_path = path.join(
    data_dir,
    f"workspace_samples_{la_min}_{la_max}_step_{step}_load_0.nc",
)
data = load_dataset(file_path)
# select data with unique mask
unique_mask = np.loadtxt(
    path.join(
        data_dir,
        f"unique_mask_{la_min}_{la_max}_step_{step}_load_0.csv",
    ),
    delimiter=",",
    dtype=bool,
)
la, r_OP, r_OP_la, A_IB = (
    data.la.data[unique_mask],
    data.r_OP.data[unique_mask],
    data.r_OP_la.data[unique_mask],
    data.A_IB.data[unique_mask],
)
# boundary data
triangles = np.loadtxt(
    path.join(
        data_dir,
        f"triangles_{la_min}_{la_max}_step_{step}_load_0.csv",
    ),
    delimiter=",",
    dtype=int,
)
sel = np.zeros(len(r_OP), bool)
for i in range(len(r_OP)):
    if i in triangles:
        sel[i] = True
r_OP_bound = r_OP[sel]

###########################
#  Cartesian Coordinates
###########################
# discete points
fig_alpha = plt.figure(f"Cartesian")
ax = fig_alpha.add_subplot(projection="3d")
ax.view_init(azim=-30)
ax.scatter(*(r_OP_bound.T) * 1000, "o", s=0.3, alpha=1, label="boundary points")
ax.set_xlabel(r"$r_{OPx}\ \mathrm{(mm)}$", labelpad=0)
ax.set_ylabel(r"$r_{OPy}\ \mathrm{(mm)}$", labelpad=0)
ax.set_zlabel(r"$r_{OPz}\ \mathrm{(mm)}$", labelpad=0)
ax.set_xticks(np.linspace(-50, 50, 5))
ax.set_yticks(np.linspace(-50, 50, 5))
ax.tick_params(axis="x", pad=0)
ax.tick_params(axis="y", pad=0)
ax.tick_params(axis="z", pad=0)
ax.set_zlim3d([87, 122])
ax.set_box_aspect([1.19047619, 1.19047619, 0.89285714])
fig_alpha.subplots_adjust(left=-0.05)


# surface
ax.view_init(azim=-30)
ax.plot_trisurf(*(r_OP * 1000).T, triangles=triangles, alpha=0.1, label="alpha shape")
# ribs
mesh = trimesh.Trimesh(vertices=r_OP, faces=triangles)
for phi in np.linspace(0, np.pi, 4, endpoint=False):
    section = mesh.section(
        plane_origin=[0, 0, 0],
        plane_normal=[np.cos(phi + np.pi / 2), np.sin(phi + np.pi / 2), 0],
    )
    if section is not None:
        [ax.plot(*(d * 1000).T, linewidth=0.5 * mm2pt) for d in section.discrete]

ax.set_xlabel(r"$x\ \mathrm{(mm)}$", labelpad=0)
ax.set_ylabel(r"$y\ \mathrm{(mm)}$", labelpad=0)
ax.set_zlabel(r"$z\ \mathrm{(mm)}$", labelpad=0)
ax.set_xticks(np.linspace(-50, 50, 5))
ax.set_yticks(np.linspace(-50, 50, 5))
ax.tick_params(axis="x", pad=0)
ax.tick_params(axis="y", pad=0)
ax.tick_params(axis="z", pad=0)
ax.set_zlim3d([87, 122])
ax.set_box_aspect([1.19047619, 1.19047619, 0.89285714])
fig_alpha.subplots_adjust(left=-0.05)
plt.legend()


la_min = 0
la_max = 50
step = 11
###########
# load data
###########
# raw data
file_path = path.join(
    data_dir,
    f"workspace_samples_{la_min}_{la_max}_step_{step}_load_0.nc",
)
data = load_dataset(file_path)
# select data with unique mask
unique_mask = np.loadtxt(
    path.join(
        data_dir,
        f"unique_mask_{la_min}_{la_max}_step_{step}_load_0.csv",
    ),
    delimiter=",",
    dtype=bool,
)
la, r_OP, r_OP_la, A_IB = (
    data.la.data[unique_mask],
    data.r_OP.data[unique_mask],
    data.r_OP_la.data[unique_mask],
    data.A_IB.data[unique_mask],
)

# alpha shape
fig = plt.figure(f"Cartesian_0", layout="constrained")
plt.subplot(2, 1, 1)
plt.scatter(
    np.linalg.norm(r_OP_bound[:, :2], axis=1) * 1000,
    r_OP_bound[:, 2] * 1000,
    s=0.3,
    alpha=1,
)
# ribs
mesh = trimesh.Trimesh(vertices=r_OP, faces=triangles)
for phi in np.linspace(0, np.pi, 4, endpoint=False):
    section = mesh.section(
        plane_origin=[0, 0, 0],
        plane_normal=[np.cos(phi + np.pi / 2), np.sin(phi + np.pi / 2), 0],
    )
    if section is not None:
        for d in section.discrete:
            rho, z, phi = cyl_coord(d)
            plt.plot(
                rho * 1000,
                z * 1000,
                linewidth=0.5 * mm2pt,
            )
# plt.xlim([-2, 72])
# plt.ylim([72, 122])
plt.grid(True)
plt.xlabel(r"$\rho\ \mathrm{(mm)}$", labelpad=0)
plt.ylabel(r"$z\ \mathrm{(mm)}$", labelpad=0)
plt.xticks(np.arange(0, 61, 10))
plt.yticks(np.arange(90, 121, 10))
#
rho, z, phi = cyl_coord(r_OP_bound)
plt.subplot(2, 1, 2)
plt.scatter(
    rho * 1000,
    np.rad2deg(phi),
    s=0.3,
    alpha=1,
)
# ribs
mesh = trimesh.Trimesh(vertices=r_OP, faces=triangles)
for phi in np.linspace(0, np.pi, 4, endpoint=False):
    section = mesh.section(
        plane_origin=[0, 0, 0],
        plane_normal=[np.cos(phi + np.pi / 2), np.sin(phi + np.pi / 2), 0],
    )
    if section is not None:
        for d in section.discrete:
            rho, z, phi = cyl_coord(d)
            plt.plot(
                rho * 1000,
                np.rad2deg(phi),
                linewidth=0.5 * mm2pt,
            )
plt.xticks(np.arange(0, 61, 10))
plt.yticks(np.arange(0, 361, 90))
plt.grid(True)
plt.xlabel(r"$\rho\ \mathrm{(mm)}$", labelpad=0)
plt.ylabel(r"$\varphi\ {(^\circ)}$", labelpad=0)
plt.show()
