from os import path, walk
import numpy as np
from matplotlib import pyplot as plt
from xarray import load_dataset


data_dir = "Experimental_data"


def find_files(root_folder, starting=(), ending=()):
    """Find files with specified endings within a root folder"""
    video_files = []
    # Walk through all folders and subfolders in the root directory
    for dirpath, dirnames, filenames in walk(root_folder):
        for file in filenames:
            # Check if the file ends with .mcap extension
            match = True
            if len(starting) and not file.startswith(starting):
                match = False
            if len(ending) and not file.endswith(ending):
                match = False
            if match:
                video_files.append(path.join(dirpath, file))

    return sorted(video_files)



def sync_exp_data(
    data_cam, data_ros, t0_step=0, dt_step=0, delay_cam=0.5
):
    assert t0_step >= delay_cam
    # camera data
    data_cam["t"] += delay_cam
    stride = max(int(dt_step * data_cam["fps"]), 1)
    # slice
    id0_cam = int((t0_step - delay_cam) * data_cam["fps"])
    id1 = np.where(data_cam["t"] <= data_ros["t_force"][-1])[0][-1]
    data_cam["t"] = data_cam["t"][id0_cam:id1:stride]
    data_cam["r_OP"] = data_cam["r_OP"][id0_cam:id1:stride]
    data_cam["A_IB"] = data_cam["A_IB"][id0_cam:id1:stride]
    data_cam["psi_IB"] = data_cam["psi_IB"][id0_cam:id1:stride]
    # ros data interpolation
    t_eval = data_cam["t"]
    data_ros["forces"] = np.array(
        [
            np.interp(t_eval, data_ros["t_force"], data_ros["forces"][:, i])
            for i in range(data_ros["forces"].shape[1])
        ]
    ).T
    data_ros["t_force"] = t_eval
    data_ros["motor_angle"] = np.array(
        [
            np.interp(t_eval, data_ros["t_motor"], data_ros["motor_angle"][:, i])
            for i in range(data_ros["motor_angle"].shape[1])
        ]
    ).T
    data_ros["t_force"] = t_eval


def C_I_r_OP(x, y, z, eps=1e-8, angle_singular=0):
    # https://en.wikipedia.org/wiki/Cylindrical_coordinate_system
    r = np.sqrt(x**2 + y**2)
    if r < eps:
        phi = angle_singular
    elif y >= 0:
        phi = np.arccos(x / r)
    else:
        phi = np.pi * 2 - np.arccos(x / r)
    return np.array([r, z, phi])


###########
# load data
###########
# sampling data
la_min = 0
la_max = 50
step = 11
# raw data
file_path = path.join(
    data_dir,
    "Workspace",
    f"workspace_samples_{la_min}_{la_max}_step_{step}_load_0.nc",
)
data = load_dataset(file_path)
# select data with unique mask
unique_mask = np.loadtxt(
    path.join(
        data_dir,
        "Workspace",
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
        "Workspace",
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


###########
# load data
###########
r_OP_traj = []
r_OP_cam = []
r_OP_pred = []
for exp_dir in [
    "traj_lin_R5_theta2",
    "traj_lin_R10_theta5",
    "traj_lin_R15_theta7",
    "traj_lin_R20_theta10",
    "traj_lin_R25_theta13",
    "traj_lin_R30_theta15",
    "traj_lin_R35_theta18",
    "traj_lin_R40_theta21",
    "traj_lin_R45_theta24",
    "traj_lin_R50_theta28",
]:
    data_dir_ = path.join(data_dir, "Circular_Motion", exp_dir)
    data_traj = np.load(find_files(data_dir_, "traj_", ".npy")[0], allow_pickle=True).item()
    data_ros = np.load(find_files(data_dir_, "rosbag2_", ".npy")[0], allow_pickle=True).item()
    data_cam = np.load(find_files(data_dir_, "Cam", "_track.npy")[0], allow_pickle=True).item()
    data_cam["t"] = np.arange(data_cam["r_OP"].shape[0]) / data_cam["fps"]
    sync_exp_data(data_cam, data_ros, t0_step=1, dt_step=2, delay_cam=0.5)
    data_ros["forces"][0] *= 0

    ##################
    # model simulation
    ##################
    # param = ModelParameter()
    # param.E_A = param.E_I = 7.07287431e5
    # param.G_A = param.G_J = 2.28672004e5
    # model = S1T4ForceParallel(param=param)
    # r_OP, A_IB = model.apply_forces(data_ros.forces, eval_keys=["r_OP", "A_IB"], verbose= True)
    data_pred = np.load(find_files(data_dir_, "pred_", ".npy")[0], allow_pickle=True).item()
    data_cam["r_OP"] += data_pred["r_OP"][0] - data_cam["r_OP"][0]
    data_cam["A_IB"] = data_pred["A_IB"][0].T @ (data_cam["A_IB"][0].T @ data_cam["A_IB"])

    #
    r_OP_traj.append(data_traj["r_OP"][15:105])
    r_OP_cam.append(data_cam["r_OP"][15:105])
    r_OP_pred.append(data_pred["r_OP"][15:105])

r_OP_traj = np.array(r_OP_traj)
r_OP_cam = np.array(r_OP_cam)
r_OP_pred = np.array(r_OP_pred)


##################################
#  Circular motion in 3D workspace
##################################
error = np.max(np.abs(r_OP_traj - r_OP_cam), axis=2)
# I_r_OP
fig_alpha = plt.figure(f"3D")
ax = fig_alpha.add_subplot(projection="3d")
ax.view_init(azim=-30)
# ax.scatter(*(I_r_OP.T) * 1000, "o", color = 'grey', s=1, alpha=1, label = 'samples')
ax.plot_trisurf(*(r_OP * 1000).T, triangles=triangles, alpha=0.1)

scatter = ax.scatter(
    *(r_OP_traj.reshape((-1, 3)).T) * 1000,
    "or",
    s=2,
    alpha=1,
    c=error.flatten() * 1000,
    cmap="coolwarm",
)
# surface
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
# plt.legend()
# plt.tight_layout(pad=2)
plt.colorbar(scatter, label=r"$\Delta r_{OP}$ [mm]")
# plt.tight_layout(pad=3)
###########
# last circular motion
######################
t = data_ros["t_force"][15:-15] - data_ros["t_force"][15]
fig = plt.figure(f"Circualr", layout="constrained")
ax = fig.subplots(4, 1, sharex=True)
ax[1].plot(t, data_traj["forces"][15:-15] - data_ros["forces"][15:-15], "o")
ax[2].plot(t, data_cam["r_OP"][15:-15] * 1000, "o", zorder=4)
ax[2].plot(t, data_traj["r_OP"][15:-15] * 1000, "-k")
ax[3].plot(t, data_traj["r_OP"][15:-15] * 1000 - data_cam["r_OP"][15:-15] * 1000, "o")
ax[0].plot(t, data_ros["forces"][15:-15], "o", zorder=4)
ax[0].plot(t, data_traj["forces"][15:-15], "-k")
ax[0].legend(
    [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$", "ref"], loc="right"
)
ax[2].legend([r"$x$", r"$y$", r"$z$", "ref"], loc="right")
ax[3].set_xticks(np.arange(0, 181, 30))
ax[3].set_xlabel(r"$t\ \mathrm{(s)}$", labelpad=0)
ax[0].set_ylabel(r"$\lambda$ (N)")
ax[1].set_ylabel(r"$\Delta\lambda$ (N)")
ax[2].set_ylabel(r"$\mathbf{r}_{OP}$ (mm)")
ax[3].set_ylabel(r"$\Delta\mathbf{r}_{OP}$ (mm)")

for a in ax:
    a.grid()
plt.show()