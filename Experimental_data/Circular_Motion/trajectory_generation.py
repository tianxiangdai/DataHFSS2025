from scipy.optimize import minimize
from matplotlib import pyplot as plt
from os import path
import numpy as np
import scipy
from cardillo.math import Exp_SO3
from tdcrobots.math import I_S_r_OP, I_S_v_P, lin_interpolate
from tdcrobots.models import S1T4ForceParallel, ModelParameter
from tdcrobots.dataio import DataSim

nt = 4


def gen_traj(r_min, theta_max):
    param = ModelParameter()
    param.E_A = param.E_I = 7.07287431e5
    param.G_A = param.G_J = 2.28672004e5
    # param.r_OP0_marker_platform = np.array(
    #     [-4.46762703e-04,  2.09797511e-04, param.h0_marker_platform], float
    # )
    # param.AIB0_marker_platform = Exp_SO3(np.array([0, 0, 1.53540118e-02]))
    model = S1T4ForceParallel(param=param)
    r_OP = model.apply_forces(np.zeros(4), verbose=False, eval_keys=["r_OP"])

    print("=======> Trajectory Generation")
    # t, r, theta, phi
    t1 = int(theta_max / np.deg2rad(2))
    t1 = 15
    desired = np.array(
        [
            [0, r_OP[2], 0, 0],
            [t1, r_min, theta_max, 0],
            [t1 + 90, r_min, theta_max, np.pi * 2],
            [2 * t1 + 90, r_OP[2], 0, np.pi * 2],
        ]
    )
    traj_fun = lambda t: lin_interpolate(t, desired[:, 0], desired[:, 1:])
    t_eval = np.arange(desired[-1, 0] + 1)
    # traj_interp = np.array([traj_fun(t)[0] for t in t_eval])
    # plt.figure()
    # plt.plot(t_eval, traj_interp)
    # plt.show()
    traj = []
    x0 = np.ones(nt) * 0
    for t in t_eval:
        (r, theta, phi), (dr, dtheta, dphi) = traj_fun(t)
        r_OP = I_S_r_OP(r, theta, phi, degree=False)
        I_v_P_desired = I_S_v_P(r, theta, phi, dr, dtheta, dphi, degree=False)

        x_scale = 1e5

        # --------------------
        #   Force Minimization
        # --------------------
        def con(x):
            # model simulation
            force_list = x * x_scale
            r_OP_model = model.apply_forces(
                force_list, eval_keys=["r_OP"], verbose=True, force_steps=1
            )
            # error
            err = r_OP_model - r_OP
            return err

        def jac(x):
            # model simulation
            force_list = x * x_scale
            r_OP_la = model.apply_forces(
                force_list, eval_keys=["r_OP_la"], verbose=True, force_steps=1
            )
            return r_OP_la * x_scale

        cons = {"type": "eq", "fun": con, "jac": jac}

        sol = minimize(
            # lambda x: x@x / 2,
            # x0,
            # jac = lambda x: x,
            # #
            lambda x: (x - x0) @ (x - x0) / 2,
            x0,
            jac=lambda x: (x - x0),
            #
            # lambda x: x@np.array([1, 1, 1, 1]),
            # x0,
            # jac = lambda x: np.array([1, 1, 1, 1]),
            #
            # lambda x: (x - 25/x_scale) @ (x - 25/x_scale)/2,
            # lambda x: (x - 25/x_scale) @ np.sign(x - 25/x_scale),
            # jac = lambda x: (x-x0),
            # jac = lambda x: np.sign(x - 25/x_scale),
            # hess = lambda x: np.eye(4),
            options={"disp": False},
            bounds=[(0, 50 / x_scale)] * 4,
            # tol=1e-6,
            constraints=cons,
            method="SLSQP",
        )
        if not sol.success:
            print(sol)
            break
        print(
            "r %.2f" % (r * 1000),
            "theta %.2f" % (theta),
            "phi %.2f" % (phi),
            "forces",
            sol.x * x_scale,
        )
        traj.append(
            np.array(
                (
                    t,
                    r,
                    theta,
                    phi,
                    dr,
                    dtheta,
                    dphi,
                    *r_OP,
                    *I_v_P_desired,
                    *sol.x * x_scale,
                )
            )
        )
        x0 = sol.x
    traj = np.array(traj)

    return {
        "t": traj[:, 0],
        "sph_coord": traj[:, 1:4],
        "r_OP": traj[:, 7:10],
        "forces": traj[:, 13:17],
    }


np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)


r0 = 0.121
r1 = 0.10744
theta0 = 0
theta1 = np.deg2rad(28.84)
k = (r1 - r0) / (theta1 - theta0)
R_max = 0.05
num_traj = 10
for i in range(num_traj):
    R = R_max / num_traj * (i + 1)
    sol = scipy.optimize.root(lambda x: (r0 + k * x) * np.sin(x) - R, 0)
    theta = sol.x[0]
    r = r0 + k * theta
    z = r*np.cos(theta)
    # print(, R, theta)
    print(sol.success, R, z)
    traj = gen_traj(r, theta)
    data = DataSim()
    data.t = traj["t"]
    data.r_OP = traj["r_OP"]
    data.sph_coord = traj["sph_coord"]
    data.forces = traj["forces"]
    data.save_data(
        path.dirname(__file__),
        f"traj_lin_R{R*1000:.0f}_theta{np.rad2deg(theta):.0f}",
        prefix="",
        surfix=".npy",
    )
