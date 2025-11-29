from numpy import zeros, einsum


class Force:
    r"""Force represented w.r.t. I-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    xi : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (C) in body-fixed B-basis.
    name : str
        Name of contribution.
    """

    def __init__(self, force, subsystem, xi=zeros(3), B_r_CP=zeros(3), name="force"):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.r_OP = lambda t, q: subsystem.r_OP(t, q, xi, B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, xi, B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, xi, B_r_CP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def h(self, t):
        return self.force(t) @ self.J_P(t, self.q)

    def h_q(self, t):
        q = self.q
        return einsum("i,ijk->jk", self.force(t), self.J_P_q(t, q))
