import numpy as np
from abc import ABC
from os import path
from scipy.optimize import least_squares


from .rigid_connection import RigidConnection
from .force import Force

from .visualization import (
    VisualArUco,
    VisualRodBody,
    VisualSTL,
    VisualTendon,
    VisualCoordSystem,
)
from .solver import StaticSolver
from .math import interp1d, Exp_SO3, Log_SO3
from .system import System
from .force_line_distributed import Force_line_distributed
from .rod import Rod
from .tendon import ForceTendon


class ModelParameter:
    def __init__(self):
        self.g_accel = 9.81

        # beam
        self.m_rod = 0.433  # kg
        self.r_rod = 30e-3
        self.l_rod = 95e-3
        self.poly_degree = 3
        self.rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
        self.rod_A_IB0[0, 1] = self.rod_A_IB0[1, 2] = self.rod_A_IB0[2, 0] = 1

        self.h_rod_foot = 11.5e-3

        # marker_platform
        # self.h_marker_platform = 8.5e-3
        self.m_marker_platform = 0.185  # kg
        self.h_marker_platform = 14.5e-3  # - 0.15e-3
        self.h_marker_platform_cut = 11.5e-3
        self.h0_marker_platform = 121e-3 - self.h_marker_platform / 2

        # tendon mount hole
        self.dr_OP0_tendon_hole = np.array([0, 0, 0], dtype=np.float64)
        self.A_IB0_tendon_hole = np.eye(3, dtype=np.float64)
        self.r_hole = 65e-3

        # auxiliary variables
        self.B_r_CP_top_platform = self.rod_A_IB0.T @ np.array(
            [
                0,
                0,
                self.h_rod_foot + self.h_marker_platform - self.h_marker_platform_cut,
            ]
        )
        ##########################
        # Visualization Properties
        ##########################
        self.color_rod = (82, 108, 164)
        self.color_marker_platform = (255, 250, 240)
        self.color_tendon = (0, 200, 50)  # (130, 130, 130)
        self.color_connector = (160, 160, 160)
        self.color_ground_platform = (255, 250, 240)
        self.visual_r_tendon = 1e-3
        self.visual_len_axis_marker = 0.045
        self.visual_len_axis_rod = self.r_rod
        self.visual_rod_centerline_nelement = 2
        self.visual_rod_nonlinear_subdivision = 4
        self.visual_rod_body_nelement = 2**self.visual_rod_nonlinear_subdivision
        self.visual_rod_opacity = 0.5
        self.visual_marker_platform_opacity = 1
        self.visual_marker_opacity = 1
        self.visual_connector_opacity = 1
        self.visual_tendon_opacity = 1

        ###############
        # Rod Stiffness
        ###############
        self.E = 7.07287431e5
        self.G = 2.28672004e5


class __ModelBase(ABC):
    def __init__(self, param: ModelParameter) -> None:
        self.param = param
        self.visual_twins = []

        # ---- system ----
        self.system = System()

        # ---- rod ----
        radius = param.r_rod
        area = np.pi * radius**2
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        second_moment = np.diag([2, 1, 1]) / 4 * np.pi * radius**4

        EA = param.E * area
        EI = param.E * second_moment[1, 1]
        GA = param.G * area
        GJ = param.G * second_moment[0, 0]
        Ei = np.array([EA, GA, GA])
        Fi = np.array([GJ, EI, EI])

        # generate initial configuration
        r_OP0 = np.array([0, 0, param.h_rod_foot])
        m_rod = param.m_rod
        m_platform = param.m_marker_platform
        g = param.g_accel
        L0_rod = param.l_rod / (1 - (m_platform + m_rod / 2) * g / EA)
        Q = Rod.straight_configuration(
            param.poly_degree,
            L0_rod,
            r_OP0,
            self.param.rod_A_IB0,
        )

        def r_OP(xi):
            z = (
                xi + m_rod * g / EA / 2 * (xi**2 - 2 * xi) - m_platform * g / EA * xi
            ) * L0_rod
            return np.array([z, 0, 0], dtype=np.float64)

        A_IB = lambda xi: np.eye(3, dtype=np.float64)
        q0 = Rod.pose_configuration(
            param.poly_degree,
            r_OP,
            A_IB,
            r_OP0=r_OP0,
            A_IB0=self.param.rod_A_IB0,
        )

        self.rod = Rod(
            Ei,
            Fi,
            param.poly_degree,
            Q=Q,
            q0=q0,
        )

        # ---- rigid connections ----
        rc1 = RigidConnection(self.rod, self.system.origin, xi1=0)

        # ---- external forces ----
        gravity_rod = Force_line_distributed(
            np.array([0, 0, -m_rod * g / L0_rod]),
            self.rod,
        )
        gravity_marker_platform = Force(
            np.array([0, 0, -m_platform * g]),
            self.rod,
            B_r_CP=self.param.rod_A_IB0.T
            @ np.array(
                [
                    0,
                    0,
                    param.h_rod_foot
                    + param.h_marker_platform / 2
                    - param.h_marker_platform_cut,
                ]
            ),
            name="gravity_marker_platform",
            xi=1,
        )

        # ---- add to system ----
        self.system.add(self.rod)
        self.system.add(gravity_marker_platform)
        self.system.add(gravity_rod)
        self.system.add(rc1)

    def assemble(self):
        self.system.assemble()
        self.force_init = np.array([td.la(0) for td in self.tendons])
        # ---- solver ----
        self.static_solver = StaticSolver(
            self.system,
            n_load_steps=1,
            verbose=False,
            # options=SolverOptions(continue_with_unconverged=False),
        )
        self.nt = -1
        self.__init_visualization()

    def set_new_initial_state(self, q0, u0, t0=None, **assemble_kwargs):
        self.system.set_new_initial_state(q0, u0, t0, **assemble_kwargs)
        self.static_solver.renew_initial_state()

    def __init_visualization(self):
        param = self.param
        self.visual_twins.append(
            VisualRodBody(
                self.rod,
                param.r_rod,
                param.visual_rod_body_nelement,
                param.visual_rod_nonlinear_subdivision,
                opacity=param.visual_rod_opacity,
                color=param.color_rod,
            )
        )
        for i in range(self.rod.nnodes):
            self.visual_twins.append(
                VisualCoordSystem(
                    self.rod, param.visual_len_axis_rod, xi=i / (self.rod.nnodes - 1)
                )
            )
        self.visual_twins.append(
            VisualSTL(
                self.rod,
                path.join(path.dirname(__file__), "stl", "Segment_Foot_V2.stl"),
                xi=1,
                scale=1e-3,
                A_BM=self.param.rod_A_IB0.T,
                B_r_CP=self.param.rod_A_IB0.T
                @ np.array(
                    [
                        0,
                        0,
                        param.h_rod_foot / 2,
                    ]
                ),
                color=param.color_connector,
                opacity=param.visual_connector_opacity,
            )
        )
        self.visual_twins.append(
            VisualSTL(
                self.rod,
                path.join(
                    path.dirname(__file__), "stl", "Marker_Platform_Target_V2.stl"
                ),
                xi=1,
                scale=1e-3,
                A_BM=self.param.rod_A_IB0.T,
                B_r_CP=self.param.rod_A_IB0.T
                @ np.array([0, 0, param.h_rod_foot - param.h_marker_platform_cut]),
                color=param.color_marker_platform,
                opacity=param.visual_marker_platform_opacity,
            )
        )
        # self.visual_twins.append(
        #     VisualSTL(
        #         self.system.origin,
        #         path.join(path.dirname(__file__), "stl", "Ground_Platform.stl"),
        #         scale=1e-3,
        #         A_BM=np.diag(np.array([-1, 1, -1], dtype=np.float64),
        #         color=param.color_ground_platform,
        #     )
        # )
        self.visual_twins.append(
            VisualSTL(
                self.system.origin,
                path.join(path.dirname(__file__), "stl", "Segment_Foot_V2.stl"),
                scale=1e-3,
                B_r_CP=np.array([0, 0, param.h_rod_foot / 2]),
                color=param.color_connector,
                opacity=param.visual_connector_opacity,
            )
        )
        self.visual_twins.append(
            VisualCoordSystem(
                self.rod,
                param.visual_len_axis_marker,
                xi=1,
                A_BM=self.param.rod_A_IB0.T,
                B_r_CP=self.param.B_r_CP_top_platform,
                opacity=param.visual_marker_platform_opacity,
            )
        )
        self.visual_twins.append(
            VisualArUco(
                self.rod,
                xi=1,
                mk_size=0.04,
                mk_dis=0.05,
                A_BM=self.param.rod_A_IB0.T,
                B_r_CP=self.param.B_r_CP_top_platform,
                opacity=param.visual_marker_opacity,
            )
        )
        for tendon in self.tendons:
            self.visual_twins.append(
                VisualTendon(
                    tendon,
                    r_tendon=param.visual_r_tendon,
                    color=param.color_tendon,
                    opacity=param.visual_tendon_opacity,
                )
            )
            # self.visual_twins[-1].actors[0].GetProperty().SetSpecular(0.8)

    def apply_forces(
        self,
        forces: np.ndarray,
        eval_keys=[],
        verbose=False,
        force_steps=1,
        ret_all_steps=False,
        warm_start=True,
    ):
        forces = np.atleast_2d(forces)

        ts = np.linspace(0, 1, forces.shape[0] + 1)
        # -----------
        #   tendons
        # -----------
        _forces = np.vstack((self.force_init, forces))
        for i, tendon in enumerate(self.tendons):
            tendon.set_force(lambda t, i=i: interp1d(ts, _forces[:, i], t))
        # ------------
        #   Solve
        # ------------
        self.static_solver.set_load_steps(forces.shape[0] * force_steps)
        self.static_solver.verbose = verbose
        self.sol = self.static_solver.solve(warm_start=warm_start)
        # ------------------------
        #   Solution Evaluation
        # ------------------------
        if ret_all_steps:
            t, q, x = self.sol.t[1:], self.sol.q[1:], self.static_solver.x[1:]
        else:
            t, q, x = (
                self.sol.t[force_steps::force_steps],
                self.sol.q[force_steps::force_steps],
                self.static_solver.x[force_steps::force_steps],
            )
        if warm_start:
            self.force_init = forces[-1]
        return self.__data_evaluation(t, q, eval_keys)

    def __data_evaluation(self, t, q, eval_keys):
        nt = len(t)
        if nt != self.nt:
            self.nt = nt
            ntendon = len(self.tendons)
            self.r_OP = np.empty((nt, 3), dtype=np.float64)
            self.A_IB = np.empty((nt, 3, 3), dtype=np.float64)
            self.l_tendon = np.empty((nt, ntendon), dtype=np.float64)
            self.la_tendon = np.empty((nt, ntendon), dtype=np.float64)

        last_node = self.rod
        for i, ti, qi in zip(range(nt), t, q):
            # displacement x,y,z
            if "r_OP" in eval_keys:
                self.r_OP[i] = last_node.r_OP(
                    ti, qi[last_node.qDOF], B_r_CP=self.param.B_r_CP_top_platform, xi=1
                )
            # transformation matrix
            if "A_IB" in eval_keys:
                self.A_IB[i] = (
                    last_node.A_IB(ti, qi[last_node.qDOF]) @ self.param.rod_A_IB0.T
                )
            # tendon length
            if "l_tendon" in eval_keys:
                self.l_tendon[i] = [
                    tendon.l(ti, qi[tendon.qDOF]) for tendon in self.tendons
                ]
            # tendon force
            if "la_tendon" in eval_keys:
                self.la_tendon[i] = [
                    tendon.la(
                        ti,
                    )
                    for tendon in self.tendons
                ]

        evals = []
        for key in eval_keys:
            if nt == 1 and not key == "sol":
                evals.append(np.squeeze(getattr(self, key), axis=0))
            else:
                evals.append(getattr(self, key))
        return evals[0] if len(evals) == 1 else tuple(evals)

    def apply_poses(
        self,
        poses: np.ndarray,
        tendon_activations=np.array([1]),
        la_bounds=np.array([0, np.inf]),
        eval_keys=[],
        verbose=False,
        warm_start=True,
    ):
        poses = np.atleast_2d(poses)
        print(tendon_activations.shape)
        if tendon_activations.ndim == 1:
            if len(tendon_activations) == 1:
                tendon_activations = np.tile(
                    tendon_activations, (poses.shape[0], len(self.tendons))
                )
            else:
                tendon_activations = np.tile(tendon_activations, (poses.shape[0], 1))
        print(tendon_activations.shape)
        print(la_bounds.shape)
        if la_bounds.ndim == 1:
            la_bounds = np.tile(la_bounds, (poses.shape[0], len(self.tendons), 1))
        elif la_bounds.ndim == 2:
            la_bounds = np.tile(la_bounds, (poses.shape[0], 1))
        print(la_bounds.shape)

        eval_keys_ext = list(set(["r_OP", "A_IB"] + eval_keys))

        # --------------------
        #   Force Minimization
        # --------------------
        def sim(la, la_act, pos, eval_keys=[]):
            f = np.zeros_like(la_act, dtype=np.float64)
            f[la_act == 1] = la
            ret = self.__apply_forces_statics(
                f,
                eval_keys=eval_keys_ext,
                verbose=False,
                force_steps=1,
                warm_start=warm_start,
            )
            A_IB_meas = Exp_SO3(pos[3:])
            r_OP = ret[eval_keys_ext.index("r_OP")]
            A_IB = ret[eval_keys_ext.index("A_IB")]
            ret = [ret[eval_keys_ext.index(k)] for k in eval_keys]
            # error
            err = np.array(
                (
                    *(r_OP - pos[:3]) * 1000,
                    *np.rad2deg(Log_SO3(A_IB_meas.T @ A_IB)),
                )
            )
            return (err, *ret)

        x_scale = 1e4
        x0 = self.force_init[tendon_activations[0] == 1] / x_scale
        force_opt = np.zeros((len(poses), self.nt), dtype=np.float64)
        for i, pos, la_act, la_bd in zip(
            range(len(poses)), poses, tendon_activations, la_bounds
        ):
            if not warm_start:
                x0 = self.force_init[la_act == 1] / x_scale
            x_bd = la_bd[la_act == 1] / x_scale
            x0 = np.minimum(np.maximum(x0, x_bd[:, 0]), x_bd[:, 1])
            sol = least_squares(
                lambda x: sim(x * x_scale, la_act, pos)[0],
                x0,
                bounds=x_bd.T,
            )
            if not sol.success:
                print(sol)
                break
            force_opt[i, la_act == 1] = sol.x * x_scale
            if verbose:
                print(
                    i,
                    "pos",
                    np.array2string(
                        pos,
                        formatter={"float_kind": lambda x: "%.5f" % x},
                        separator=", ",
                    ),
                    "forces",
                    np.array2string(
                        force_opt,
                        formatter={"float_kind": lambda x: "%.1f" % x},
                        separator=", ",
                    ),
                    f"pos err {round(np.linalg.norm(sol.fun[:3]), 2)} [mm]",
                    f"ori err {round(np.linalg.norm(sol.fun[3:]), 2)} [deg]",
                )
            if warm_start:
                x0 = sol.x
        return self.__apply_forces_statics(
            force_opt,
            eval_keys=eval_keys,
            verbose=False,
            force_steps=1,
            warm_start=warm_start,
        )


class S1T4ForceParallel(__ModelBase):
    def __init__(self, param=ModelParameter()) -> None:
        super().__init__(param)
        B_r_CP_lists = [
            [
                np.array([param.r_hole * np.cos(phi), param.r_hole * np.sin(phi), 0]),
                param.rod_A_IB0.T
                @ (
                    param.dr_OP0_tendon_hole
                    + param.A_IB0_tendon_hole
                    @ np.array(
                        [
                            param.r_hole * np.cos(phi),
                            param.r_hole * np.sin(phi),
                            param.h_rod_foot - param.h_marker_platform_cut,
                        ]
                    )
                ),
            ]
            for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
        ]
        # ---- tendons ----
        self.tendons = []
        for B_r_CP_list in B_r_CP_lists:
            tendon = ForceTendon(
                subsystem_list=[self.system.origin, self.rod],
                connectivity=[(0, 1)],
                B_r_CP_list=B_r_CP_list,
                xi_list=[0, 1],
            )
            self.tendons.append(tendon)
        self.system.add(*self.tendons)
        self.assemble()


class S1T4ForceCrossCW(__ModelBase):
    def __init__(self, param=ModelParameter()) -> None:
        super().__init__(param)
        B_r_CP_lists = [
            [
                np.array([param.r_hole * np.cos(phi), param.r_hole * np.sin(phi), 0]),
                param.rod_A_IB0.T
                @ (
                    param.dr_OP0_tendon_hole
                    + param.A_IB0_tendon_hole
                    @ np.array(
                        [
                            param.r_hole * np.cos(phi + np.pi / 2),
                            param.r_hole * np.sin(phi + np.pi / 2),
                            param.h_rod_foot - param.h_marker_platform_cut,
                        ]
                    )
                ),
            ]
            for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
        ]
        # ---- tendons ----
        self.tendons = []
        for B_r_CP_list in B_r_CP_lists:
            tendon = ForceTendon(
                subsystem_list=[self.system.origin, self.rod],
                connectivity=[(0, 1)],
                B_r_CP_list=B_r_CP_list,
                xi_list=[0, 1],
            )
            self.tendons.append(tendon)
        self.system.add(*self.tendons)
        self.assemble()


class S1T4ForceCrossCCW(__ModelBase):
    def __init__(self, param=ModelParameter()) -> None:
        super().__init__(param)
        B_r_CP_lists = [
            [
                np.array([param.r_hole * np.cos(phi), param.r_hole * np.sin(phi), 0]),
                param.rod_A_IB0.T
                @ (
                    param.dr_OP0_tendon_hole
                    + param.A_IB0_tendon_hole
                    @ np.array(
                        [
                            param.r_hole * np.cos(phi - np.pi / 2),
                            param.r_hole * np.sin(phi - np.pi / 2),
                            param.h_rod_foot - param.h_marker_platform_cut,
                        ]
                    )
                ),
            ]
            for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
        ]
        # ---- tendons ----
        self.tendons = []
        for B_r_CP_list in B_r_CP_lists:
            tendon = ForceTendon(
                subsystem_list=[self.system.origin, self.rod],
                connectivity=[(0, 1)],
                B_r_CP_list=B_r_CP_list,
                xi_list=[0, 1],
            )
            self.tendons.append(tendon)
        self.system.add(*self.tendons)
        self.assemble()
