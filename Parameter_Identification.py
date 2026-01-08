from os import path, walk
import numpy as np
from scipy.optimize import least_squares
from itertools import product

from mylibs.math import Log_SO3

from mylibs.models import (
    S1T4ForceParallel,
    S1T4ForceCrossCW,
    S1T4ForceCrossCCW,
    ModelParameter,
)
from matplotlib import pyplot as plt

data_dir = "Experimental_data/Parameter_Identification"

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


def load_data(data_dir, shift_origin=True):
    # data_dir = path.join(path.dirname(__file__), data_dir)
    # find data files
    file_ros = find_files(data_dir, "rosbag2_", ".npy")[0]
    file_pose = find_files(data_dir, "Cam1", "_track.npy")[0]

    # ros data
    data_ros = np.load(file_ros, allow_pickle=True).item()
    t_force = data_ros["data_force_reading"][:, 0] / 1e9
    forces = data_ros["data_force_reading"][:, 1:]
    for i in range(1, 5):
        if str(i) not in path.basename(data_dir):
            forces[:, i - 1] *= 0
    (
        t_motor,
        motor_angle,
        motor_vel,
        motor_pwm,
        motor_current,
        motor_angle_traj,
        motor_vel_traj,
    ) = np.split(data_ros["data_motor_states"], [1, 5, 9, 13, 17, 21], axis=1)
    t_motor = t_motor.flatten() / 1e9
    # motion tracking
    data_pose = np.load(file_pose, allow_pickle=True).item()
    fps = data_pose["fps"]
    A_IB = data_pose["A_IB"]
    r_OP = data_pose["r_OP"]
    if shift_origin:
        A_IB = A_IB @ A_IB[0].T
        r_OP = r_OP - r_OP[0] + np.array([0, 0, 0.121])
        psi_IB = np.array([Log_SO3(A) for A in A_IB])
    t_pose = np.arange(r_OP.shape[0]) / fps
    # interpolate force data
    force_interp = np.array(
        [np.interp(t_pose, t_force, forces[:, i]) for i in range(forces.shape[1])]
    ).T
    motor_angle_interp = np.array(
        [
            np.interp(t_pose, t_motor, motor_angle[:, i])
            for i in range(motor_angle.shape[1])
        ]
    ).T

    # slice data
    stride = 2 * fps
    t_pose_slice = t_pose[::stride]
    r_OP_slice = r_OP[::stride]
    psi_IB_slice = psi_IB[::stride]
    A_IB_slice = A_IB[::stride]
    t_force_slice = t_pose[fps // 2 :: stride]  # due to delay of camera
    force_slice = force_interp[fps // 2 :: stride]
    t_motor_slice = t_pose[fps // 2 :: stride]  # due to delay of camera
    motor_angle_slice = motor_angle_interp[fps // 2 :: stride]
    # if shift_origin:
    #     force_slice[0] = 0
    return (
        t_force_slice,
        force_slice,
        t_pose_slice,
        r_OP_slice,
        psi_IB_slice,
        A_IB_slice,
        t_motor_slice,
        motor_angle_slice,
    )


def combine_data(data_dirs, cycles=None, shift_origin=True):
    force_list, r_OP_list, psi_IB_list, A_IB_list, motor_angle_list = [], [], [], [], []
    for data_dir in data_dirs:
        (
            t_force_slice,
            force_slice,
            t_pose_slice,
            r_OP_slice,
            psi_IB_slice,
            A_IB_slice,
            t_motor_slice,
            motor_angle_slice,
        ) = load_data(data_dir, shift_origin=shift_origin)
        force_list.append(force_slice[: 22 * cycles])
        r_OP_list.append(r_OP_slice[: 22 * cycles])
        psi_IB_list.append(psi_IB_slice[: 22 * cycles])
        A_IB_list.append(A_IB_slice[: 22 * cycles])
        motor_angle_list.append(motor_angle_slice[: 22 * cycles])
    force_list = np.concatenate(force_list)
    r_OP_list = np.concatenate(r_OP_list)
    psi_IB_list = np.concatenate(psi_IB_list)
    A_IB_list = np.concatenate(A_IB_list)
    motor_angle_list = np.concatenate(motor_angle_list)
    return force_list, r_OP_list, psi_IB_list, A_IB_list, motor_angle_list


def eval_measurement(r_OP, A_IB, force, ModelClass, r_hole, h_marker_platform):
    if ModelClass is S1T4ForceParallel:
        r_PM1 = [el @ np.array([r_hole, 0, -h_marker_platform]) for el in A_IB]
        r_PM2 = [el @ np.array([0, r_hole, -h_marker_platform]) for el in A_IB]
        r_PM3 = [el @ np.array([-r_hole, 0, -h_marker_platform]) for el in A_IB]
        r_PM4 = [el @ np.array([0, -r_hole, -h_marker_platform]) for el in A_IB]
    elif ModelClass is S1T4ForceCrossCW:
        r_PM1 = [el @ np.array([0, r_hole, -h_marker_platform]) for el in A_IB]
        r_PM2 = [el @ np.array([-r_hole, 0, -h_marker_platform]) for el in A_IB]
        r_PM3 = [el @ np.array([0, -r_hole, -h_marker_platform]) for el in A_IB]
        r_PM4 = [el @ np.array([r_hole, 0, -h_marker_platform]) for el in A_IB]
    elif ModelClass is S1T4ForceCrossCCW:
        r_PM1 = [el @ np.array([0, -r_hole, -h_marker_platform]) for el in A_IB]
        r_PM2 = [el @ np.array([r_hole, 0, -h_marker_platform]) for el in A_IB]
        r_PM3 = [el @ np.array([0, r_hole, -h_marker_platform]) for el in A_IB]
        r_PM4 = [el @ np.array([-r_hole, 0, -h_marker_platform]) for el in A_IB]
    n1 = [np.array([r_hole, 0, 0]) - r_PM1[i] - r_OP[i] for i in range(A_IB.shape[0])]
    n2 = [np.array([0, r_hole, 0]) - r_PM2[i] - r_OP[i] for i in range(A_IB.shape[0])]
    n3 = [np.array([-r_hole, 0, 0]) - r_PM3[i] - r_OP[i] for i in range(A_IB.shape[0])]
    n4 = [np.array([0, -r_hole, 0]) - r_PM4[i] - r_OP[i] for i in range(A_IB.shape[0])]
    n1 = [el / np.linalg.norm(el) for el in n1]
    n2 = [el / np.linalg.norm(el) for el in n2]
    n3 = [el / np.linalg.norm(el) for el in n3]
    n4 = [el / np.linalg.norm(el) for el in n4]
    # resulting force
    F_P = [
        n1[i] * force[i, 0]
        + n2[i] * force[i, 1]
        + n3[i] * force[i, 2]
        + n4[i] * force[i, 3]
        for i in range(force.shape[0])
    ]

    M_P = [
        np.cross(
            r_PM1[i],
            n1[i] * force[i, 0],
        )
        + np.cross(
            r_PM2[i],
            n2[i] * force[i, 1],
        )
        + np.cross(
            r_PM3[i],
            n3[i] * force[i, 2],
        )
        + np.cross(
            r_PM4[i],
            n4[i] * force[i, 3],
        )
        for i in range(A_IB.shape[0])
    ]
    return n1, n2, n3, n4, F_P, M_P


def fit_EA_compression(r_OP_comp, A_IB_comp, la_t_comp, model_param: ModelParameter):
    n1_comp, n2_comp, n3_comp, n4_comp, F_P_comp, M_P_comp = eval_measurement(
        r_OP_comp,
        A_IB_comp,
        la_t_comp,
        S1T4ForceParallel,
        model_param.r_hole,
        model_param.h_marker_platform,
    )
    Fz_comp = np.array(F_P_comp)[:, 2]
    eps = (
        r_OP_comp[:, 2]
        - model_param.h0_marker_platform
        - model_param.h_marker_platform / 2
    ) / model_param.l_rod

    # lienar regression
    EA = np.dot(eps, Fz_comp) / np.dot(eps, eps)
    return EA, eps, Fz_comp


def fit_GJ_torsion(
    r_OP_cw, A_IB_cw, force_cw, r_OP_ccw, A_IB_ccw, force_ccw, model_param
):
    n1_cw, n2_cw, n3_cw, n4_cw, F_P_cw, M_P_cw = eval_measurement(
        r_OP_cw,
        A_IB_cw,
        force_cw,
        S1T4ForceCrossCW,
        model_param.r_hole,
        model_param.h_marker_platform,
    )
    n1_ccw, n2_ccw, n3_ccw, n4_ccw, F_z_ccw, M_P_ccw = eval_measurement(
        r_OP_ccw,
        A_IB_ccw,
        force_ccw,
        S1T4ForceCrossCCW,
        model_param.r_hole,
        model_param.h_marker_platform,
    )
    tau_z = np.array(M_P_cw + M_P_ccw)[:, 2]
    dpsi_z = (
        np.array([Log_SO3(A) for A in A_IB_cw] + [Log_SO3(A) for A in A_IB_ccw])[:, 2]
        / model_param.l_rod
    )
    dpsi_z -= dpsi_z[0]

    # lienar regression
    GJ = np.dot(dpsi_z, tau_z) / np.dot(dpsi_z, dpsi_z)
    return GJ, dpsi_z, tau_z


def run_identification(
    model_param: ModelParameter,
    la_t_meas,
    r_OP_meas,
    A_IB_meas,
    psi_IB_meas,
    tendon_activations,
    split_index,
    x0=None,
    optimize=True,
    diff_step=1e-5,
):
    # split measurement
    la_t_comp, la_t_cw, la_t_ccw, la_t_bend = np.split(la_t_meas, split_index)
    r_OP_comp, r_OP_cw, r_OP_ccw, r_OP_bend = np.split(r_OP_meas, split_index)
    A_IB_comp, A_IB_cw, A_IB_ccw, A_IB_bend = np.split(A_IB_meas, split_index)
    psi_comp, psi_cw, psi_ccw, psi_bend = np.split(psi_IB_meas, split_index)
    t_act_comp, t_act_cw, t_act_ccw, t_act_bend = np.split(
        tendon_activations, split_index
    )

    # error function
    def objfun(x):
        # set model parameters
        (
            model_param.E_A,
            model_param.E_I,
            model_param.G_J,
        ) = x
        model_param.G_A = model_param.G_J
        # model simulation
        r_OP_model = []
        A_IB_model = []
        psi_IB_model = []
        la_model = []
        err = []
        for la_t, ModelClass, r_OP, A_IB, psi, t_act in zip(
            (la_t_comp, la_t_cw, la_t_ccw, la_t_bend),
            [S1T4ForceParallel, S1T4ForceCrossCW, S1T4ForceCrossCCW, S1T4ForceParallel],
            [r_OP_comp, r_OP_cw, r_OP_ccw, r_OP_bend],
            [A_IB_comp, A_IB_cw, A_IB_ccw, A_IB_bend],
            [psi_comp, psi_cw, psi_ccw, psi_bend],
            [t_act_comp, t_act_cw, t_act_ccw, t_act_bend],
        ):
            model = ModelClass(model_param)
            if la_t.shape[0] == 0:
                continue
            r_OP_model_i, A_IB_model_i = model.apply_forces(
                la_t,
                verbose=True,
                eval_keys=["r_OP", "A_IB"],
            )
            la_model.extend(la_t)
            psi_IB_model_i = np.array([Log_SO3(el) for el in A_IB_model_i])
            r_OP_model.extend(r_OP_model_i)
            A_IB_model.extend(A_IB_model_i)
            psi_IB_model.extend(psi_IB_model_i)
            err_i = np.hstack(
                (
                    (r_OP - r_OP_model_i) * 1000,
                    np.rad2deg([Log_SO3(A.T @ B) for A, B in zip(A_IB, A_IB_model_i)]),
                    # np.rad2deg(psi_i - psi_IB_model_i),
                )
            ).flatten()
            err.extend(err_i)
        r_OP_model = np.array(r_OP_model)
        A_IB_model = np.array(A_IB_model)
        psi_IB_model = np.array(psi_IB_model)
        la_model = np.array(la_model)
        err = np.hstack(err)
        print(
            np.array2string(
                x, formatter={"float_kind": lambda x: "%.5e" % x}, separator=", "
            )
        )
        return (
            err,
            r_OP_model,
            A_IB_model,
            psi_IB_model,
            la_model,
        )

    if optimize:
        x_scale = np.array([1e5] * 3)
        r = least_squares(
            lambda x: objfun(x)[0],
            x0=x0,
            method="lm",
            x_scale=x_scale,
            diff_step=diff_step,
        )
        x_opt = r.x
        print(r)
    else:
        x_opt = x0
    return (
        x_opt,
        *objfun(x_opt),
    )



###########
# Load Data
###########
# Experiments in sequence of compression, torsion clockwise, torsion counter clockwise, bending.
exp_dirs = (
    "Exp_compression_cable_13",
    # "Exp_compression_cable_24",
    "Exp_torsion_cable_13_CW",
    # "Exp_torsion_cable_24_CW",
    "Exp_torsion_cable_13_CCW",
    # "Exp_torsion_cable_24_CCW",
    "Exp_bend_cable_1",
    "Exp_bend_cable_2",
    "Exp_bend_cable_3",
    "Exp_bend_cable_4",
)
exp_dirs = [
    path.join(
        data_dir,
        dir,
    )
    for dir in exp_dirs
]
la_t_meas, r_OP_meas, psi_IB_meas, A_IB_meas, motor_angle_meas = combine_data(
    exp_dirs, cycles=1, shift_origin=True
)
n_per_exp = la_t_meas.shape[0] // len(exp_dirs)
split_index = [n_per_exp * 1, n_per_exp * 1, n_per_exp * 1]
split_index = np.cumsum(split_index)
tendon_activations = np.vstack(
    [[1, 0, 1, 0]] * n_per_exp
    + [[0, 1, 0, 1]] * n_per_exp
    + [[1, 0, 1, 0]] * n_per_exp
    + [[0, 1, 0, 1]] * n_per_exp
    + [[1, 0, 1, 0]] * n_per_exp
    + [[0, 1, 0, 1]] * n_per_exp
    + [[1, 0, 0, 0]] * n_per_exp
    + [[0, 1, 0, 0]] * n_per_exp
    + [[0, 0, 1, 0]] * n_per_exp
    + [[0, 0, 0, 1]] * n_per_exp
)

# split measurement
la_t_comp, la_t_cw, la_t_ccw, la_t_bend = np.split(la_t_meas, split_index)
r_OP_comp, r_OP_cw, r_OP_ccw, r_OP_bend = np.split(r_OP_meas, split_index)
A_IB_comp, A_IB_cw, A_IB_ccw, A_IB_bend = np.split(A_IB_meas, split_index)
psi_comp, psi_cw, psi_ccw, psi_bend = np.split(psi_IB_meas, split_index)


##########################
# Parameter Identification
##########################
model_param = ModelParameter()
model_param.rod_nelement = 1
model_param.poly_degree = 3
# linear fit_EA
EA0, eps, Fz = fit_EA_compression(r_OP_comp, A_IB_comp, la_t_comp, ModelParameter())
E_A0 = EA0 / (np.pi * model_param.r_rod**2)
# linear fit_GJ
GJ0, dpsi_z, tau_z = fit_GJ_torsion(
    r_OP_cw, A_IB_cw, la_t_cw, r_OP_ccw, A_IB_ccw, la_t_ccw, ModelParameter()
)
G_J0 = GJ0 * 2 / (np.pi * model_param.r_rod**4)
print("initial guess E_A:", E_A0)
print("initial guess G_J:", G_J0)
# fmt: off
# model fitting
x0 = np.array([7.07291e+05, 7.07291e+05, 2.26539e+05], dtype=np.float64) 
# fmt: on
diff_step = 1e-3
optimize = False
(x_opt, err, r_OP_model, A_IB_model, psi_IB_model, la_t_model) = run_identification(
    model_param,
    la_t_meas,
    r_OP_meas,
    A_IB_meas,
    psi_IB_meas,
    tendon_activations,
    split_index,
    x0=x0,
    optimize=optimize,
    diff_step=diff_step,
)
err = err.reshape((-1, 6))
print(f"Error {np.linalg.norm(err):.3f}")


# split measurement
la_t_comp_model, la_t_cw_model, la_t_ccw_model, la_t_bend_model = np.split(
    la_t_model, split_index
)
r_OP_comp_model, r_OP_cw_model, r_OP_ccw_model, r_OP_bend_model = np.split(
    r_OP_model, split_index
)
A_IB_comp_model, A_IB_cw_model, A_IB_ccw_model, A_IB_bend_model = np.split(
    A_IB_model, split_index
)
psi_comp_model, psi_cw_model, psi_ccw_model, psi_bend_model = np.split(
    psi_IB_model, split_index
)

###########
# plot data
###########
split_index = np.hstack((split_index, split_index[-1] + np.array([22, 44, 66])))

for (
    k,
    err_k,
    r_OP_meas_k,
    A_IB_meas_k,
    psi_IB_meas_k,
    la_t_meas_k,
    r_OP_model_k,
    A_IB_model_k,
    psi_IB_model_k,
    la_t_model_k,
    motor_angle_k,
    name,
) in zip(
    range(7),
    np.split(err, split_index),
    np.split(r_OP_meas, split_index),
    np.split(A_IB_meas, split_index),
    np.split(psi_IB_meas, split_index),
    np.split(la_t_meas, split_index),
    np.split(r_OP_model, split_index),
    np.split(A_IB_model, split_index),
    np.split(psi_IB_model, split_index),
    np.split(la_t_model, split_index),
    np.split(motor_angle_meas, split_index),
    [
        "param_identification_Compression",
        "param_identification_Torsion_CW",
        "param_identification_Torsion_CCW",
        "param_identification_Bend1",
        "param_identification_Bend2",
        "param_identification_Bend3",
        "param_identification_Bend4",
    ],
):
    err_pose = np.hstack(
        (
            (r_OP_meas_k - r_OP_model_k) * 1000,
            np.array(
                [
                    np.rad2deg(Log_SO3(A.T @ B))
                    for A, B in zip(A_IB_meas_k, A_IB_model_k)
                ]
            ),
        )
    )
    if not len(la_t_meas_k):
        continue
    # print error
    print(name)
    print(
        f"\t Average position error: {np.round(np.mean(np.abs(err_pose[:,:3]), axis=0), 3)} (mm)",
    )
    print(
        f"\t Average rotation error: {np.round(np.mean(np.abs(err_pose[:,3:]), axis=0), 3)} [deg]",
    )
    print(
        f"\t Maximal position error: {np.round(np.max(np.abs(err_pose[:,:3]), axis=0), 3)} [mm]",
    )
    print(
        f"\t Maximal rotation error: {np.round(np.max(np.abs(err_pose[:,3:]), axis=0), 3)} [deg]",
    )
    #
    f_mean = np.sum(la_t_meas_k, axis=1) / np.sum(la_t_meas_k > 0, axis=1)
    # f_sum = np.sum(la_t_model_k, axis=1)
    sel_fwd = np.array(
        [[True] * 11 + [False] * 11 for _ in range(la_t_meas_k.shape[0] // 22)]
    ).flatten()
    sel_bkwd = np.invert(sel_fwd)

    plt.figure(
        name, layout="constrained"
    )
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.plot(f_mean, r_OP_model_k[:, i] * 1000, "-r", label="model")
        plt.plot(
            f_mean[sel_fwd],
            r_OP_meas_k[sel_fwd, i] * 1000,
            "bx",
            label="forward",
            markersize=5,
        )
        plt.plot(
            f_mean[sel_bkwd], r_OP_meas_k[sel_bkwd, i] * 1000, "go", label="backward"
        )
        plt.grid()
        plt.xticks(np.arange(0, 51, 10))

    for i in range(3):
        plt.subplot(2, 3, i + 4)
        plt.plot(f_mean, np.rad2deg(psi_IB_model_k[:, i]), "-r", label="model")
        plt.plot(
            f_mean[sel_fwd],
            np.rad2deg(psi_IB_meas_k[sel_fwd, i]),
            "bx",
            label="forward",
            markersize=5,
        )
        plt.plot(
            f_mean[sel_bkwd],
            np.rad2deg(psi_IB_meas_k[sel_bkwd, i]),
            "go",
            label="backward",
        )
        plt.grid()

    # set plot
    indicator = np.zeros((2, 3))
    if "Bend1" in name:
        indicator[0, 0] = indicator[1, 1] = 1
    elif "Bend2" in name:
        indicator[0, 1] = 1
        indicator[1, 0] = -1
    elif "Bend3" in name:
        indicator[0, 0] = indicator[1, 1] = -1
    elif "Bend4" in name:
        indicator[0, 1] = -1
        indicator[1, 0] = 1
    elif "Torsion_CW" in name:
        indicator[1, 2] = -1
    elif "Torsion_CCW" in name:
        indicator[1, 2] = 1
    for i, j in product(range(2), range(3)):
        plt.subplot(2, 3, i * 3 + j + 1)
        # set limits and ticks
        if i == 0 and j == 2:
            plt.ylim([95, 145])
            plt.yticks(range(100, 145, 10))
        elif indicator[i, j] == 0:
            plt.ylim([-25, 25])
            plt.yticks(range(-20, 25, 10))
        elif indicator[i, j] == 1:
            plt.ylim([-2, 48])
            plt.yticks(range(0, 45, 10))
        elif indicator[i, j] == -1:
            plt.ylim([-48, 2])
            plt.yticks(range(-40, 5, 10))
        if i == 0:
            if j == 0:
                plt.ylabel(r"$x$ (mm)")
            elif j == 1:
                plt.ylabel(r"$y$ (mm)")
            elif j == 2:
                plt.ylabel(r"$z$ (mm)")
        else:
            if j == 0:
                plt.ylabel(r"$\psi_x$ ($^\circ$)")
            elif j == 1:
                plt.ylabel(r"$\psi_y$ ($^\circ$)")
            elif j == 2:
                plt.ylabel(r"$\psi_z$ ($^\circ$)")
        plt.xticks(np.arange(0, 51, 10))
        if i == 0:
            plt.xlabel("")
        else:
            plt.xlabel(r"$\lambda_\text{active}$ (N)")
plt.show()