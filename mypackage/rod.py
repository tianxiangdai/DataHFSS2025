import numpy as np
from scipy.special import roots_legendre
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey

from .math import Log_SO3_quat, Exp_SO3_quat, T_SO3_quat, Exp_SO3_quat_p, T_SO3_quat_P

from .math import norm, cross3, ax2skew

from .math import lagrange_basis_with_derivative

eye3 = np.eye(3, dtype=float)
eye4 = np.eye(4, dtype=float)
eye3_ext = eye3[:, None, :]
eye4_ext = eye4[:, None, :]


def gaussian_quadrature(n, interval=np.array([0, 1])):
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
        Ei,
        Fi,
        polynomial_degree,
        Q,
        *,
        q0=None,
        name=None,
    ):
        self.nelement = 1

        # Material model, Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        self.C_n = np.diag(Ei)
        self.C_m = np.diag(Fi)

        self.name = "Cosserat_rod" if name is None else name

        self.nquadrature = polynomial_degree

        # self._eval_cache = LRUCache(maxsize=nquadrature + 10)
        # self._deval_cache = LRUCache(maxsize=nquadrature + 10)

        ##############################################################
        # discretization parameters centerline (r) and orientation (p)
        ##############################################################
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
        self.quad_p, self.quad_w = gaussian_quadrature(self.nquadrature, np.array([0, 1]))

        xs = np.linspace(0, 1, self.nnodes)
        self.L_dLs = [lagrange_basis_with_derivative(xs, i) for i in range(self.nnodes)]

        # shape functions and their first derivatives
        self.N = np.array(
            [[self.L_dLs[i][0](xi) for i in range(self.nnodes)] for xi in self.quad_p]
        )
        self.N_xi = np.array(
            [[self.L_dLs[i][1](xi) for i in range(self.nnodes)] for xi in self.quad_p]
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
        self.__g_S_q = np.zeros((self.nla_S, self.nq), dtype=np.float64)
        self.__h = np.zeros(self.nu, dtype=np.float64)
        self.__h_q = np.zeros((self.nu, self.nq), dtype=np.float64)
        self.__r_OP_q = np.zeros((3, self.nq), dtype=np.float64)
        self.__r_OP_xi_q = np.zeros((3, self.nq), dtype=np.float64)
        self.__p_q = np.zeros((4, self.nq), dtype=np.float64)
        self.__p_xi_q = np.zeros((4, self.nq), dtype=np.float64)
        self.__J_P = np.zeros((3, self.nu), dtype=np.float64)
        self.__J_P_q = np.zeros((3, self.nu, self.nq), dtype=np.float64)
        self.__B_J_R = np.zeros((3, self.nu), dtype=np.float64)
        self.__B_J_R_q = np.zeros((3, self.nu, self.nq), dtype=np.float64)

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
            qp = self.quad_p[i]

            # evaluate required quantities
            _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                q, qp, N=self.N[i], N_xi=self.N_xi[i]
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
        return np.array([(L_dL[0](xi), L_dL[1](xi)) for L_dL in self.L_dLs]).T

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

    def g_S(self):
        P = self.q[self.nq_r :].reshape(4, -1)
        return np.sum(P**2, axis=0) - 1

    def g_S_q(self):
        self.__g_S_q[
            np.tile(np.arange(self.nla_S), 4), np.arange(self.nq_r, self.nq)
        ] = (2 * self.q[self.nq_r :])
        return self.__g_S_q

    def h(self, t):
        q = self.q
        self.__h.fill(0)

        for i in range(self.nquadrature):
            # extract reference state variables
            qp = self.quad_p[i]
            qw = self.quad_w[i]
            J = self.J[i]
            B_Gamma0 = self.B_Gamma0[i]
            B_Kappa0 = self.B_Kappa0[i]

            # evaluate required quantities
            _, A_IB, B_Gamma_bar, B_Kappa_bar = self._eval(
                q, qp, N=self.N[i], N_xi=self.N_xi[i]
            )

            # axial and shear strains
            B_Gamma = B_Gamma_bar / J

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m = self.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)

            ############################
            # virtual work contributions
            ############################
            n_qwi = A_IB @ B_n * qw
            for node in range(self.nnodes):
                self.__h[self.nodalDOF_r_u[node]] -= self.N_xi[i, node] * n_qwi
            m_qwi = B_m * qw
            cross = (cross3(B_Gamma_bar, B_n) + cross3(B_Kappa_bar, B_m)) * qw
            for node in range(self.nnodes):
                self.__h[self.nodalDOF_p_u[node]] += (
                    -self.N_xi[i, node] * m_qwi + self.N[i, node] * cross
                )

        return self.__h

    def h_q(self, t):
        q = self.q
        self.__h_q.fill(0)

        for i in range(self.nquadrature):
            # extract reference state variables
            qp = self.quad_p[i]
            qw = self.quad_w[i]
            J = self.J[i]
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
            ) = self._deval(q, qp, N=self.N[i], N_xi=self.N_xi[i])

            # axial and shear strains
            B_Gamma = B_Gamma_bar / J
            B_Gamma_q = B_Gamma_bar_q / J

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / J
            B_Kappa_q = B_Kappa_bar_q / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_n_B_Gamma = self.B_n_B_Gamma(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_n_B_Kappa = self.B_n_B_Kappa(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_n_q = B_n_B_Gamma @ B_Gamma_q + B_n_B_Kappa @ B_Kappa_q

            B_m = self.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m_B_Gamma = self.B_m_B_Gamma(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m_B_Kappa = self.B_m_B_Kappa(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m_q = B_m_B_Gamma @ B_Gamma_q + B_m_B_Kappa @ B_Kappa_q

            ############################
            # virtual work contributions
            ############################
            n_q_qwi = qw * (np.einsum("ikj,k->ij", A_IB_q, B_n) + A_IB @ B_n_q)
            for node in range(self.nnodes):
                self.__h_q[self.nodalDOF_r[node], :] -= self.N_xi[i, node] * n_q_qwi
            B_Gamma_bar_B_n_q_qwi = qw * (
                ax2skew(B_Gamma_bar) @ B_n_q - ax2skew(B_n) @ B_Gamma_bar_q
            )
            B_m_q_qwi = qw * B_m_q
            B_Kappa_bar_B_m_q_qwi = qw * (
                ax2skew(B_Kappa_bar) @ B_m_q - ax2skew(B_m) @ B_Kappa_bar_q
            )
            for node in range(self.nnodes):
                self.__h_q[self.nodalDOF_p_u[node], :] += (
                    self.N[i, node] * B_Gamma_bar_B_n_q_qwi
                    - self.N_xi[i, node] * B_m_q_qwi
                    + self.N[i, node] * B_Kappa_bar_B_m_q_qwi
                )

        return self.__h_q

    def r_OP(self, t, q, xi, B_r_CP=None):
        N, N_xi = self.eval_bases(xi)
        r_OC, A_IB, _, _ = self._eval(q, xi, N, N_xi)
        return r_OC if B_r_CP is None else r_OC + A_IB @ B_r_CP

    def r_OP_q(self, t, q, xi, B_r_CP=None):
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

        return (
            r_OC_q
            if B_r_CP is None
            else r_OC_q + np.einsum("ijk,j->ik", A_IB_q, B_r_CP)
        )

    def A_IB(self, t, q, xi):
        N, N_xi = self.eval_bases(xi)
        return self._eval(q, xi, N, N_xi)[1]

    def A_IB_q(self, t, q, xi):
        # return approx_fprime(q, lambda q: self.A_IB(t, q, xi))
        N, N_xi = self.eval_bases(xi)
        return self._deval(q, xi, N, N_xi)[5]

    def J_P(self, t, q, xi, B_r_CP=None):
        # evaluate required nodal shape functions
        N, _ = self.eval_bases(xi)

        # transformation matrix
        A_IB = self.A_IB(t, q, xi)

        # interpolate centerline and axis angle contributions
        self.__J_P[:, : self.nu_r] = (N[None, :, None] * eye3_ext).reshape(
            (3, -1), order="F"
        )

        if B_r_CP is None:
            self.__J_P[:, self.nu_r :].fill(0)
            return self.__J_P

        # skew symmetric matrix of B_r_CP
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde
        self.__J_P[:, self.nu_r :] = (
            -N[None, :, None] * r_CP_tilde[:, None, :]
        ).reshape((3, -1), order="F")

        return self.__J_P

    def J_P_q(self, t, q, xi, B_r_CP=None):
        # evaluate required nodal shape functions
        self.__J_P_q.fill(0)

        if B_r_CP is None:
            return self.__J_P_q

        N, _ = self.eval_bases(xi)
        A_IB_q = self.A_IB_q(t, q, xi)

        B_r_CP_tilde = ax2skew(B_r_CP)
        prod = np.einsum("ijl,jk", A_IB_q, B_r_CP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        for node in range(self.nnodes):
            nodalDOF_p_u = self.nodalDOF_p_u[node]
            self.__J_P_q[:, nodalDOF_p_u] -= N[node] * prod

        return self.__J_P_q

    def B_J_R(self, t, q, xi):
        N, _ = self.eval_bases(xi)
        self.__B_J_R[:, self.nu_r :] = (N[None, :, None] * eye3_ext).reshape(
            (3, -1), order="F"
        )
        return self.__B_J_R

    def B_J_R_q(self, t, q, xi):
        return self.__B_J_R_q

    @cachedmethod(
        lambda self: self._eval_cache,
        key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
    )
    def _eval(self, q, xi, N, N_xi):
        # evaluate shape functions
        # N, N_xi = self.basis_functions_r(xi)
        # interpolate
        q_nodes_r = q[: self.nq_r].reshape(self.nnodes, 3, order="F")
        r_OP = N @ q_nodes_r
        r_OP_xi = N_xi @ q_nodes_r

        q_nodes_p = q[self.nq_r :].reshape(self.nnodes, 4, order="F")
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
        r_nodes = q[: self.nq_r].reshape(self.nnodes, 3, order="F")
        r_OP = N @ r_nodes
        r_OP_xi = N_xi @ r_nodes

        p_nodes = q[self.nq_r :].reshape(self.nnodes, 4, order="F")
        p = N @ p_nodes
        p_xi = N_xi @ p_nodes

        self.__r_OP_q[:, : self.nq_r] = (eye3_ext * N[None, :, None]).reshape(
            (3, -1), order="F"
        )
        self.__r_OP_xi_q[:, : self.nq_r] = (eye3_ext * N_xi[None, :, None]).reshape(
            (3, -1), order="F"
        )
        self.__p_q[:, self.nq_r :] = (eye4_ext * N[None, :, None]).reshape(
            (4, -1), order="F"
        )
        self.__p_xi_q[:, self.nq_r :] = (eye4_ext * N_xi[None, :, None]).reshape(
            (4, -1), order="F"
        )

        # transformation matrix
        A_IB = Exp_SO3_quat(p, normalize=True)

        # derivative w.r.t. generalized coordinates
        A_IB_q = Exp_SO3_quat_p(p, normalize=True) @ self.__p_q

        # axial and shear strains
        B_Gamma_bar = A_IB.T @ r_OP_xi
        B_Gamma_bar_q = np.einsum("k,kij", r_OP_xi, A_IB_q) + A_IB.T @ self.__r_OP_xi_q

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
            @ self.__p_q
            + T @ self.__p_xi_q
        )

        return (
            r_OP,
            A_IB,
            B_Gamma_bar,
            B_Kappa_bar,
            self.__r_OP_q,
            A_IB_q,
            B_Gamma_bar_q,
            B_Kappa_bar_q,
        )

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
