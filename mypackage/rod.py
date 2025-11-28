import numpy as np
from scipy.special import roots_legendre
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey

from .math import Log_SO3_quat, Exp_SO3_quat, T_SO3_quat, Exp_SO3_quat_p, T_SO3_quat_P

from .math import norm, cross3, ax2skew

from .math import lagrange_basis_with_derivative

eye3 = np.eye(3, dtype=float)


def gauss(n, interval=np.array([0, 1])):
    points, weights = roots_legendre(n)

    # transfrom gauss points on new interval,
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    points = (interval[1] - interval[0]) / 2 * points + (interval[1] + interval[0]) / 2
    weights = (interval[1] - interval[0]) / 2 * weights

    return points, weights


class Rod:
    """
    displacement based static rod with one element
    """

    def __init__(
        self,
        material_model,
        polynomial_degree,
        Q,
        *,
        q0=None,
        name=None,
    ):
        self.nelement = 1

        # rod properties
        self.material_model = material_model
        self.name = "Cosserat_rod" if name is None else name

        self.nquadrature = polynomial_degree

        # self._eval_cache = LRUCache(maxsize=nquadrature + 10)
        # self._deval_cache = LRUCache(maxsize=nquadrature + 10)

        ##############################################################
        # discretization parameters centerline (r) and orientation (p)
        ##############################################################
        # TODO: combine mesh for position and orientation fields
        self.polynomial_degree = polynomial_degree

        # total number of nodes
        self.nnodes = polynomial_degree + 1

        # total number of generalized position coordinates
        self.nq_r = self.nnodes * 3
        self.nq_p = self.nnodes * 4
        self.nq = self.nq_r + self.nq_p

        # total number of generalized velocity coordinates
        self.nu_r = self.nnodes * 3
        self.nu_p = self.nnodes * 3

        # TODO: not necessary
        self.nu = self.nu_r + self.nu_p
        self.u0 = np.zeros(self.nu, dtype=float)

        # global nodal connectivity
        self.nodalDOF_r = np.arange(self.nnodes * 3).reshape(self.nnodes, 3, order="F")
        self.nodalDOF_p = (
            np.arange(self.nnodes * 4).reshape(self.nnodes, 4, order="F") + self.nq_r
        )
        self.nodalDOF_r_u = np.arange(self.nnodes * 3).reshape(
            self.nnodes, 3, order="F"
        )
        self.nodalDOF_p_u = (
            np.arange(self.nnodes * 3).reshape(self.nnodes, 3, order="F") + self.nu_r
        )

        # quadrature points and weights
        self.qp, self.qw = gauss(self.nquadrature, np.array([0, 1]))

        xs = np.linspace(0, 1, self.nnodes)
        self.L_dLs = [lagrange_basis_with_derivative(xs, i) for i in range(self.nnodes)]

        # shape functions and their first derivatives
        self.N = np.array(
            [[self.L_dLs[i][0](xi) for i in range(self.nnodes)] for xi in self.qp]
        )
        self.N_xi = np.array(
            [[self.L_dLs[i][1](xi) for i in range(self.nnodes)] for xi in self.qp]
        )

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0

        # unit quaternion constraints
        dim_g_S = 1
        self.nla_S = self.nnodes * dim_g_S
        self.nodalDOF_la_S = np.arange(self.nla_S).reshape(self.nnodes, dim_g_S)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)


        # allocate memery
        self._eval_bases_cache = LRUCache(maxsize=self.nquadrature + 20)
        self._eval_cache = LRUCache(maxsize=self.nquadrature + 20)
        self._deval_cache = LRUCache(maxsize=self.nquadrature + 20)
        self.__g_S_q = np.zeros((self.nla_S, self.nq))
        self.__h = np.zeros(self.nu, dtype=np.float64)
        self.__h_q = np.zeros((self.nu, self.nq), dtype=np.float64)
        
        self.set_reference_strains(self.Q)

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        # precompute values of the reference configuration in order to save
        # computation time J in Harsch2020b (5)
        self.J = np.zeros(self.nquadrature, dtype=float)
        # dilatation and shear strains of the reference configuration
        self.B_Gamma0 = np.zeros((self.nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.B_Kappa0 = np.zeros((self.nquadrature, 3), dtype=float)

        q = self.Q

        for i in range(self.nquadrature):
            # current quadrature point
            qpi = self.qp[i]

            # evaluate required quantities
            _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                q, qpi, N=self.N[i], N_xi=self.N_xi[i]
            )

            # length of reference tangential vector
            J = norm(B_Gamma_bar)

            # dilatation and shear strains
            B_Gamma = B_Gamma_bar / J

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / J

            # safe precomputed quantities for later
            self.J[i] = J
            self.B_Gamma0[i] = B_Gamma
            self.B_Kappa0[i] = B_Kappa


    @cachedmethod(
        lambda self: self._eval_bases_cache,
        key=lambda self, xi: hashkey(xi),
    )
    def eval_bases(self, xi):
        return np.array(
            [(L_dL[0](xi), L_dL[1](xi)) for L_dL in self.L_dLs]
        ).T

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for straight configuration."""
        nnodes = polynomial_degree + 1

        x0 = np.linspace(0, L, num=nnodes)
        y0 = np.zeros(nnodes)
        z0 = np.zeros(nnodes)
        r_OP = np.vstack((x0, y0, z0))
        for i in range(nnodes):
            r_OP[:, i] = r_OP0 + A_IB0 @ r_OP[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OP.reshape(-1, order="C")

        # extract quaternion from orientation A_IB0
        p = Log_SO3_quat(A_IB0)
        q_p = np.repeat(p, nnodes)

        return np.concatenate([q_r, q_p])

    @staticmethod
    def pose_configuration(
        polynomial_degree,
        r_OP,
        A_IB,
        xi1=1.0,
        r_OP0=np.zeros(3, dtype=np.float64),
        A_IB0=np.eye(3, dtype=np.float64),
    ):
        """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
        nnodes_r = polynomial_degree + 1

        assert callable(r_OP), "r_OP must be callable!"
        assert callable(A_IB), "A_IB must be callable!"

        xis = np.linspace(0, xi1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((3, nnodes_r))
        p0 = np.zeros((4, nnodes_r))

        for i, xii in enumerate(xis):
            r0[:, i] = r_OP0 + A_IB0 @ r_OP(xii)
            A_IBi = A_IB0 @ A_IB(xii)
            p0[:, i] = Log_SO3_quat(A_IBi)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[:, i] @ p0[:, i + 1]
            if inner < 0:
                p0[:, i + 1] *= -1

        # reshape generalized position coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")
        q_p = p0.reshape(-1, order="C")

        return np.concatenate([q_r, q_p])

    def step_callback(self, t, q, u):
        """ "Quaternion normalization after each time step."""
        for node in range(self.nnodes):
            p = q[self.nodalDOF_p[node]]
            q[self.nodalDOF_p[node]] = p / norm(p)
        return q, u

    def g_S(self, t, q):
        P = q[self.nq_r :].reshape(4, -1)
        return np.sum(P**2, axis=0) - 1

    def g_S_q(self, t, q):
        self.__g_S_q[
            np.tile(np.arange(self.nla_S), 4), np.arange(self.nq_r, self.nq)
        ] = (2 * q[self.nq_r :])
        return self.__g_S_q

    def h(self, t, q, u):
        self.__h.fill(0)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[i]
            qwi = self.qw[i]
            J = self.J[i]
            B_Gamma0 = self.B_Gamma0[i]
            B_Kappa0 = self.B_Kappa0[i]

            # evaluate required quantities
            _, A_IB, B_Gamma_bar, B_Kappa_bar = self._eval(
                q, qpi, N=self.N[i], N_xi=self.N_xi[i]
            )

            # axial and shear strains
            B_Gamma = B_Gamma_bar / J

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.material_model.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m = self.material_model.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)

            ############################
            # virtual work contributions
            ############################
            n_qwi = A_IB @ B_n * qwi
            for node in range(self.nnodes):
                self.__h[self.nodalDOF_r_u[node]] -= self.N_xi[i, node] * n_qwi
            m_qwi = B_m * qwi
            cross = (cross3(B_Gamma_bar, B_n) + cross3(B_Kappa_bar, B_m)) * qwi
            for node in range(self.nnodes):
                self.__h[self.nodalDOF_p_u[node]] += (
                    -self.N_xi[i, node] * m_qwi + self.N[i, node] * cross
                )

        return self.__h

    def h_q(self, t, q, u):
        self.__h_q.fill(0)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[i]
            qwi = self.qw[i]
            Ji = self.J[i]
            B_Gamma0 = self.B_Gamma0[i]
            B_Kappa0 = self.B_Kappa0[i]

            # evaluate required quantities
            (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_q,
                A_IB_q,
                B_Gamma_bar_q,
                B_Kappa_bar_q,
            ) = self._deval(q, qpi, N=self.N[i], N_xi=self.N_xi[i])

            # axial and shear strains
            B_Gamma = B_Gamma_bar / Ji
            B_Gamma_q = B_Gamma_bar_q / Ji

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / Ji
            B_Kappa_q = B_Kappa_bar_q / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.material_model.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_n_B_Gamma = self.material_model.B_n_B_Gamma(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_n_B_Kappa = self.material_model.B_n_B_Kappa(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_n_q = B_n_B_Gamma @ B_Gamma_q + B_n_B_Kappa @ B_Kappa_q

            B_m = self.material_model.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m_B_Gamma = self.material_model.B_m_B_Gamma(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_m_B_Kappa = self.material_model.B_m_B_Kappa(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_m_q = B_m_B_Gamma @ B_Gamma_q + B_m_B_Kappa @ B_Kappa_q

            ############################
            # virtual work contributions
            ############################
            n_q_qwi = qwi * (np.einsum("ikj,k->ij", A_IB_q, B_n) + A_IB @ B_n_q)
            for node in range(self.nnodes):
                self.__h_q[self.nodalDOF_r[node], :] -= self.N_xi[i, node] * n_q_qwi
            B_Gamma_bar_B_n_q_qwi = qwi * (
                ax2skew(B_Gamma_bar) @ B_n_q - ax2skew(B_n) @ B_Gamma_bar_q
            )
            B_m_q_qwi = qwi * B_m_q
            B_Kappa_bar_B_m_q_qwi = qwi * (
                ax2skew(B_Kappa_bar) @ B_m_q - ax2skew(B_m) @ B_Kappa_bar_q
            )
            for node in range(self.nnodes):
                self.__h_q[self.nodalDOF_p_u[node], :] += (
                    self.N[i, node] * B_Gamma_bar_B_n_q_qwi
                    - self.N_xi[i, node] * B_m_q_qwi
                    + self.N[i, node] * B_Kappa_bar_B_m_q_qwi
                )

        return self.__h_q

    def r_OP(self, t, q, xi, B_r_CP=np.zeros(3, dtype=float)):
        N, N_xi = self.eval_bases(xi)
        r_OC, A_IB, _, _ = self._eval(q, xi, N, N_xi)
        return r_OC + A_IB @ B_r_CP

    def r_OP_q(self, t, q, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required quantities
        N, N_xi = self.eval_bases(xi)
        (
            r_OC,
            A_IB,
            _,
            _,
            r_OC_q,
            A_IB_q,
            _,
            _,
        ) = self._deval(q, xi, N, N_xi)

        return r_OC_q + np.einsum("ijk,j->ik", A_IB_q, B_r_CP)

    def A_IB(self, t, q, xi):
        N, N_xi = self.eval_bases(xi)
        return self._eval(q, xi, N, N_xi)[1]

    def A_IB_q(self, t, q, xi):
        # return approx_fprime(q, lambda q: self.A_IB(t, q, xi))
        N, N_xi = self.eval_bases(xi)
        return self._deval(q, xi, N, N_xi)[5]

    def J_P(self, t, q, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.eval_bases(xi)

        # transformation matrix
        A_IB = self.A_IB(t, q, xi)

        # skew symmetric matrix of B_r_CP
        B_r_CP_tilde = ax2skew(B_r_CP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nu), dtype=q.dtype)
        for node in range(self.nnodes):
            J_P[:, self.nodalDOF_r_u[node]] += N[node] * eye3
        r_CP_tilde = A_IB @ B_r_CP_tilde
        for node in range(self.nnodes):
            J_P[:, self.nodalDOF_p_u[node]] -= N[node] * r_CP_tilde

        return J_P

    def J_P_q(self, t, q, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.eval_bases(xi)

        B_r_CP_tilde = ax2skew(B_r_CP)
        A_IB_q = self.A_IB_q(t, q, xi)
        prod = np.einsum("ijl,jk", A_IB_q, B_r_CP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nu, self.nq), dtype=float)
        for node in range(self.nnodes):
            nodalDOF_p_u = self.nodalDOF_p_u[node]
            J_P_q[:, nodalDOF_p_u] -= N[node] * prod

        return J_P_q

    def B_J_R(self, t, q, xi):
        N_p, _ = self.eval_bases(xi)
        B_J_R = np.zeros((3, self.nu), dtype=float)
        for node in range(self.nnodes):
            B_J_R[:, self.nodalDOF_p_u[node]] += N_p[node] * eye3
        return B_J_R

    def B_J_R_q(self, t, q, xi):
        return np.zeros((3, self.nu, self.nq), dtype=float)

    @cachedmethod(
        lambda self: self._eval_cache,
        key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
    )
    def _eval(self, q, xi, N, N_xi):
        # evaluate shape functions
        # N, N_xi = self.basis_functions_r(xi)
        # interpolate
        q_nodes_r = q[:self.nq_r].reshape(self.nnodes, 3, order="F")
        r_OP = N @ q_nodes_r
        r_OP_xi = N_xi @ q_nodes_r

        q_nodes_p = q[self.nq_r:].reshape(self.nnodes, 4, order="F")
        p = N @ q_nodes_p
        p_xi = N_xi @ q_nodes_p

        # transformation matrix
        A_IB = Exp_SO3_quat(p, normalize=True)

        # dilatation and shear strains
        B_Gamma_bar = A_IB.T @ r_OP_xi

        # curvature, Rucker2018 (17)
        B_Kappa_bar = T_SO3_quat(p, normalize=True) @ p_xi

        return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

    @cachedmethod(
        lambda self: self._deval_cache,
        key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
    )
    def _deval(self, q, xi, N, N_xi):
        # evaluate shape functions
        # N, N_xi = self.basis_functions_r(xi)

        # interpolate
        r_nodes = q[:self.nq_r].reshape(self.nnodes, 3, order="F")
        r_OP = N @ r_nodes
        r_OP_xi = N_xi @ r_nodes

        p_nodes = q[self.nq_r:].reshape(self.nnodes, 4, order="F")
        p = N @ p_nodes
        p_xi = N_xi @ p_nodes
        
        r_OP_q = np.zeros((3, self.nq), dtype=q.dtype)
        r_OP_xi_q = np.zeros((3, self.nq), dtype=q.dtype)
        p_q = np.zeros((4, self.nq), dtype=q.dtype)
        p_xi_q = np.zeros((4, self.nq), dtype=q.dtype)

        for node in range(self.nnodes):
            nodalDOF_r = self.nodalDOF_r[node]
            r_OP_q[:, nodalDOF_r] += N[node] * np.eye(3, dtype=float)
            r_OP_xi_q[:, nodalDOF_r] += N_xi[node] * np.eye(3, dtype=float)
            
            nodalDOF_p = self.nodalDOF_p[node]
            p_q[:, nodalDOF_p] += N[node] * np.eye(4, dtype=float)
            p_xi_q[:, nodalDOF_p] += N_xi[node] * np.eye(4, dtype=float)
        
        # r_OP_q[:, :self.nq_r] = np.concatenate(np.eye(3, dtype=float)[:, :, None] * N[None, None, :], axis=1)
        # r_OP_xi_q[:, :self.nq_r] = np.concatenate(np.eye(3, dtype=float)[:, :, None] * N_xi[None, None, :], axis=1)
        # p_q[:, self.nq_r:] = np.concatenate(np.eye(4, dtype=float)[:, :, None] * N[None, None, :], axis=1)
        # p_xi_q[:, self.nq_r:] = np.concatenate(np.eye(4, dtype=float)[:, :, None] * N_xi[None, None, :], axis=1)

        # transformation matrix
        A_IB = Exp_SO3_quat(p, normalize=True)

        # derivative w.r.t. generalized coordinates
        A_IB_q = Exp_SO3_quat_p(p, normalize=True) @ p_q

        # axial and shear strains
        B_Gamma_bar = A_IB.T @ r_OP_xi
        B_Gamma_bar_q = np.einsum("k,kij", r_OP_xi, A_IB_q) + A_IB.T @ r_OP_xi_q

        # curvature, Rucker2018 (17)
        T = T_SO3_quat(p, normalize=True)
        B_Kappa_bar = T @ p_xi

        # B_Kappa_bar_q = approx_fprime(q, lambda q: self._eval(q, xi)[3])
        B_Kappa_bar_q = (
            np.einsum(
                "ijk,j->ik",
                T_SO3_quat_P(p, normalize=True),
                p_xi,
            )
            @ p_q
            + T @ p_xi_q
        )

        return (
            r_OP,
            A_IB,
            B_Gamma_bar,
            B_Kappa_bar,
            r_OP_q,
            A_IB_q,
            B_Gamma_bar_q,
            B_Kappa_bar_q,
        )



class Simo1986:
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable rod with quadratic strain energy
        function found in Simo1986 (2.8), (2.9) and (2.10).

        Parameters
        ----------
        Ei : np.ndarray (3,)
            E0: dilatational stiffness, i.e., rigidity with resepct to volumetric change.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """

        self.Ei = Ei
        self.Fi = Fi

        self.C_n = np.diag(self.Ei)
        self.C_m = np.diag(self.Fi)

        self.C_n_inv = np.linalg.inv(self.C_n)
        self.C_m_inv = np.linalg.inv(self.C_m)

    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dG = B_Gamma - B_Gamma0
        return self.C_n @ dG

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0
        return self.C_m @ dK

    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return self.C_n

    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return self.C_m
