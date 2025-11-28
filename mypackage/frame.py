import numpy as np


class Frame:
    def __init__(
        self,
        r_OP=np.zeros(3),
        A_IB=np.eye(3),
        name="frame",
    ):
        """Frame parameterized by time dependent position and orientation.

        Parameters
        ----------
        r_OP : np.array(3) (callable/non-callable)
            Frame position.
        A_IB : np.array(3, 3) (callable/non-callable)
            Frame orientation.
        name : str
            Name of frame.
        """
        self.r_OP__ = r_OP if callable(r_OP) else lambda t: r_OP
        self.A_IB__ = r_OP if callable(r_OP) else lambda t: A_IB

        self.nq = 0
        self.nu = 0
        self.q0 = np.array([])
        self.u0 = np.array([])

        self.name = name

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.array([], dtype=int)

    def local_uDOF_P(self, xi=None):
        return np.array([], dtype=int)

    def A_IB(self, t, q=None, xi=None):
        return self.A_IB__(t)

    def A_IB_q(self, t, q=None, xi=None):
        return np.array([]).reshape((3, 3, 0))

    def r_OP(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return self.r_OP__(t) + self.A_IB__(t) @ B_r_CP

    def r_OP_q(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def B_J_R(self, t, q, xi=None):
        return np.array([]).reshape((3, 0))

    def B_J_R_q(self, t, q=None, xi=None):
        return np.array([]).reshape((3, 0, 0))
