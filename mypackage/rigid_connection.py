import numpy as np

from .math import ax2skew, cross3


def concatenate_qDOF(object):
    qDOF1 = object.subsystem1.qDOF
    qDOF2 = object.subsystem2.qDOF

    object.qDOF = np.concatenate((qDOF1, qDOF2))
    object._nq1 = nq1 = len(qDOF1)
    object._nq2 = len(qDOF2)
    object._nq = object._nq1 + object._nq2

    return qDOF1, qDOF2


def concatenate_uDOF(object):
    uDOF1 = object.subsystem1.uDOF
    uDOF2 = object.subsystem2.uDOF

    object.uDOF = np.concatenate((uDOF1, uDOF2))
    object._nu1 = nu1 = len(uDOF1)
    object._nu2 = len(uDOF2)
    object._nu = object._nu1 + object._nu2

    return uDOF1, uDOF2


class RigidConnection:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        projection_pairs_rotation=[(1, 2), (2, 0), (0, 1)],
        xi1=None,
        xi2=None,
        name="rigid_connection",
    ):
        self.name = name
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2

        # guard against flawed constrained_axes input
        self.nla_g_rot = len(projection_pairs_rotation)
        for pair in projection_pairs_rotation:
            assert len(np.unique(pair)) == 2
            for i in pair:
                assert i in [0, 1, 2]

        self.nla_g = 3 + self.nla_g_rot
        self.projection_pairs = projection_pairs_rotation

        self.constrain_orientation = self.nla_g_rot > 0

    def assembler_callback(self):
        qDOF1, qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[qDOF1], self.xi1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[qDOF2], self.xi2
        )

        # check for A_IB of subsystem 1
        A_IB10 = self.subsystem1.A_IB(
            self.subsystem1.t0, self.subsystem1.q0[qDOF1], self.xi1
        )

        self.r_OJ0 = r_OP10

        self.A_IJ0 = A_IB10

        # self.B1_r_P1J0 = A_IB10.T @ (self.r_OJ0 - r_OP10)
        self.B1_r_P1J0 = None
        A_K1J0 = A_IB10.T @ self.A_IJ0

        # check for A_IB of subsystem 2
        A_IB20 = self.subsystem2.A_IB(
            self.subsystem2.t0, self.subsystem2.q0[qDOF2], self.xi2
        )

        self.B2_r_P2J0 = A_IB20.T @ (self.r_OJ0 - r_OP20)
        A_K2J0 = A_IB20.T @ self.A_IJ0

        self.auxiliary_functions(A_K1J0, A_K2J0)

    def auxiliary_functions(
        self,
        A_K1B0=None,
        A_K2B0=None,
    ):
        nq1 = self._nq1
        nu1 = self._nu1

        # auxiliary functions for subsystem 1
        self.r_OJ1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.xi1, self.B1_r_P1J0
        )
        self.r_OJ1_q1 = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.xi1, self.B1_r_P1J0
        )
        self.J_J1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.xi1, self.B1_r_P1J0
        )
        self.J_J1_q1 = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.xi1, self.B1_r_P1J0
        )
        self.A_IJ1 = lambda t, q: self.subsystem1.A_IB(t, q[:nq1], self.xi1) @ A_K1B0
        self.A_IJ1_q1 = lambda t, q: np.einsum(
            "ijl,jk->ikl", self.subsystem1.A_IB_q(t, q[:nq1], self.xi1), A_K1B0
        )
        self.J_R1 = lambda t, q: self.subsystem1.A_IB(
            t, q[:nq1], self.xi1
        ) @ self.subsystem1.B_J_R(t, q[:nq1], self.xi1)
        self.J_R1_q1 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem1.A_IB_q(t, q[:nq1], self.xi1),
            self.subsystem1.B_J_R(t, q[:nq1], self.xi1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IB(t, q[:nq1], self.xi1),
            self.subsystem1.B_J_R_q(t, q[:nq1], self.xi1),
        )

        # auxiliary functions for subsystem 2
        self.r_OJ2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.xi2, self.B2_r_P2J0
        )
        self.r_OJ2_q2 = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.xi2, self.B2_r_P2J0
        )
        self.J_J2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.xi2, self.B2_r_P2J0
        )
        self.J_J2_q2 = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.xi2, self.B2_r_P2J0
        )
        self.A_IJ2 = lambda t, q: self.subsystem2.A_IB(t, q[nq1:], self.xi2) @ A_K2B0
        self.A_IJ2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk", self.subsystem2.A_IB_q(t, q[nq1:], self.xi2), A_K2B0
        )
        self.J_R2 = lambda t, q: self.subsystem2.A_IB(
            t, q[nq1:], self.xi2
        ) @ self.subsystem2.B_J_R(t, q[nq1:], self.xi2)
        self.J_R2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem2.A_IB_q(t, q[nq1:], self.xi2),
            self.subsystem2.B_J_R(t, q[nq1:], self.xi2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IB(t, q[nq1:], self.xi2),
            self.subsystem2.B_J_R_q(t, q[nq1:], self.xi2),
        )

    def g(self, t):
        q = self.q
        g = np.zeros(self.nla_g, dtype=q.dtype)
        g[:3] = self.r_OJ2(t, q) - self.r_OJ1(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            for i, (a, b) in enumerate(self.projection_pairs):
                g[3 + i] = A_IJ1[:, a] @ A_IJ2[:, b]

        return g

    def g_q(self, t):
        q = self.q
        nq1 = self._nq1
        g_q = np.zeros((self.nla_g, self._nq), dtype=q.dtype)

        g_q[:3, :nq1] = -self.r_OJ1_q1(t, q)
        g_q[:3, nq1:] = self.r_OJ2_q2(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                g_q[3 + i, :nq1] = A_IJ2[:, b] @ A_IJ1_q1[:, a]
                g_q[3 + i, nq1:] = A_IJ1[:, a] @ A_IJ2_q2[:, b]

        return g_q

    def W_g(self, t):
        q = self.q
        nu1 = self._nu1
        W_g = np.zeros((self._nu, self.nla_g), dtype=q.dtype)
        W_g[:nu1, :3] = -self.J_J1(t, q).T
        W_g[nu1:, :3] = self.J_J2(t, q).T

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

            for i, (a, b) in enumerate(self.projection_pairs):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                W_g[:, 3 + i] = n @ J

        return W_g

    def Wla_g_q(self, t):
        q = self.q
        la_g = self.la_g
        nq1 = self._nq1
        nu1 = self._nu1
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=np.common_type(q, la_g))

        Wla_g_q[:nu1, :nq1] += np.einsum("i,ijk->jk", -la_g[:3], self.J_J1_q1(t, q))
        Wla_g_q[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[:3], self.J_J2_q2(t, q))

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            J_R1 = self.J_R1(t, q)
            J_R2 = self.J_R2(t, q)
            J_R1_q1 = self.J_R1_q1(t, q)
            J_R2_q2 = self.J_R2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                n_q1 = -ax2skew(e_b) @ A_IJ1_q1[:, a]
                n_q2 = ax2skew(e_a) @ A_IJ2_q2[:, b]
                Wla_g_q[:nu1, :nq1] += np.einsum(
                    "i,ijk->jk", la_g[3 + i] * n, J_R1_q1
                ) + np.einsum("ij,ik->kj", la_g[3 + i] * n_q1, J_R1)
                Wla_g_q[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3 + i] * n_q2, J_R1)
                Wla_g_q[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3 + i] * n_q1, J_R2)
                Wla_g_q[nu1:, nq1:] += np.einsum(
                    "i,ijk->jk", -la_g[3 + i] * n, J_R2_q2
                ) + np.einsum("ij,ik->kj", -la_g[3 + i] * n_q2, J_R2)

        return Wla_g_q
