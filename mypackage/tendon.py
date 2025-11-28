
import numpy as np

from .math import norm

class nPointInteraction:
    def __init__(
        self,
        subsystem_list,
        connectivity,
        xi_list=None,
        B_r_CP_list=None,
    ) -> None:
        self.subsystems = subsystem_list
        self.n_subsystems = len(subsystem_list)
        self.xis = self.n_subsystems * [(0,)] if xi_list is None else xi_list
        self.Bi_r_CPis = (
            self.n_subsystems * [np.zeros(3)] if B_r_CP_list is None else B_r_CP_list
        )
        self.connectivity = connectivity

    def assembler_callback(self):
        self._nq: list[int] = []
        self._nu: list[int] = []

        self.qDOF = np.array([], dtype=int)
        self.uDOF = np.array([], dtype=int)

        for sys, xi in zip(self.subsystems, self.xis):
            self._nq.append(len(self.qDOF))
            self.qDOF = np.concatenate([self.qDOF, sys.qDOF])

            self._nu.append(len(self.uDOF))
            self.uDOF = np.concatenate([self.uDOF, sys.uDOF])
        self._nq.append(len(self.qDOF))
        self._nu.append(len(self.uDOF))
        self._nq = np.array(self._nq, int)
        self._nu = np.array(self._nu, int)

        self.nq_val = [
            np.arange(*self._nq[i : i + 2]) for i in range(self.n_subsystems)
        ]

        self.nu_val = [
            np.arange(*self._nu[i : i + 2]) for i in range(self.n_subsystems)
        ]

    def r_OPk(self, t, q, k):
        return self.subsystems[k].r_OP(
            t, q[self.nq_val[k]], self.xis[k], self.Bi_r_CPis[k]
        )

    def r_OPk_qk(self, t, q, k):
        return self.subsystems[k].r_OP_q(
            t, q[self.nq_val[k]], self.xis[k], self.Bi_r_CPis[k]
        )

    def v_Pk(self, t, q, u, k):
        return self.subsystems[k].v_P(
            t,
            q[self.nq_val[k]],
            u[self.nu_val[k]],
            self.xis[k],
            self.Bi_r_CPis[k],
        )

    def v_Pk_qk(self, t, q, u, k):
        return self.subsystems[k].v_P_q(
            t,
            q[self.nq_val[k]],
            u[self.nu_val[k]],
            self.xis[k],
            self.Bi_r_CPis[k],
        )

    def J_Pk(self, t, q, k):
        return self.subsystems[k].J_P(
            t, q[self.nq_val[k]], self.xis[k], self.Bi_r_CPis[k]
        )

    def J_Pk_qk(self, t, q, k):
        return self.subsystems[k].J_P_q(
            t, q[self.nq_val[k]], self.xis[k], self.Bi_r_CPis[k]
        )

    def r_PiPj(self, t, q, i, j):
        return self.r_OPk(t, q, j) - self.r_OPk(t, q, i)

    def _nij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        l = norm(r_PiPj)
        return r_PiPj / l

    def _nij_qij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        gij = norm(r_PiPj)
        tmp = np.outer(r_PiPj, r_PiPj) / (gij**3)
        r_OPi_qi = self.r_OPk_qk(t, q, i)
        r_OPj_qj = self.r_OPk_qk(t, q, j)
        n_qi = -r_OPi_qi / gij + tmp @ r_OPi_qi
        n_qj = r_OPj_qj / gij - tmp @ r_OPj_qj
        return n_qi, n_qj

    def l(self, t, q):
        g = 0
        for i, j in self.connectivity:
            g += norm(self.r_OPk(t, q, j) - self.r_OPk(t, q, i))
        return g

    def l_q(self, t, q):
        g_q = np.zeros((self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            g_q[self.nq_val[i]] += -nij @ self.r_OPk_qk(t, q, i)
            g_q[self.nq_val[j]] += nij @ self.r_OPk_qk(t, q, j)
        return g_q

    def l_dot(self, t, q, u):
        gamma = 0
        for i, j in self.connectivity:
            gamma += self._nij(t, q, i, j) @ (
                self.v_Pk(t, q, u, j) - self.v_Pk(t, q, u, i)
            )
        return gamma

    def l_dot_q(self, t, q, u):
        gamma_q = np.zeros((self._nq[-1]), dtype=np.common_type(q, u))
        for i, j in self.connectivity:
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            nij = self._nij(t, q, i, j)
            vi, vj = self.v_Pk(t, q, u, i), self.v_Pk(t, q, u, j)
            gamma_q[self.nq_val[i]] += (vj - vi) @ nij_qi - nij @ self.v_Pk_qk(
                t, q, u, i
            )
            gamma_q[self.nq_val[j]] += (vj - vi) @ nij_qj - nij @ self.v_Pk_qk(
                t, q, u, j
            )
        return gamma_q

    def W_l(self, t, q):
        W = np.zeros((self._nu[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            W[self.nu_val[i]] += -self.J_Pk(t, q, i).T @ nij
            W[self.nu_val[j]] += self.J_Pk(t, q, j).T @ nij
        return W

    def W_l_q(self, t, q):
        W_q = np.zeros((self._nu[-1], self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nui, nui1, nuj, nuj1 = self._nu[[i, i + 1, j, j + 1]]
            nqi, nqi1, nqj, nqj1 = self._nq[[i, i + 1, j, j + 1]]
            nij = self._nij(t, q, i, j)
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            J_Pi = self.J_Pk(t, q, i)
            J_Pj = self.J_Pk(t, q, j)
            W_q[nui:nui1, nqi:nqi1] += (
                -self.J_Pk_qk(t, q, i).T @ nij
            ).T - J_Pi.T @ nij_qi
            W_q[nuj:nuj1, nqj:nqj1] += (
                self.J_Pk_qk(t, q, j).T @ nij
            ).T + J_Pj.T @ nij_qj
            W_q[nui:nui1, nqj:nqj1] += -J_Pi.T @ nij_qj
            W_q[nuj:nuj1, nqi:nqi1] += J_Pj.T @ nij_qi
        return W_q


class ForceTendon(nPointInteraction):
    def __init__(
        self,
        subsystem_list,
        connectivity,
        xi_list=None,
        B_r_CP_list=None,
    ) -> None:
        super().__init__(subsystem_list, connectivity, xi_list, B_r_CP_list)

    def h(self, t, q, u):
        return -self.la(t) * self.W_l(t, q)

    def h_q(self, t, q, u):
        return -self.la(t) * self.W_l_q(t, q)

    def la(self, t):
        return 0.0

    def set_force(self, force):
        self.la = force if callable(force) else lambda t: force
        