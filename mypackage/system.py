import numpy as np

from .frame import Frame

properties = []

properties.extend(["h", "h_q"])

properties.extend(["g"])

properties.extend(["g_S"])

properties.extend(["assembler_callback", "step_callback"])


class System:
    """Sparse model implementation which assembles all global objects without
    copying on body and element level.

    Parameters
    ----------
    t0 : float
        Initial time of the initial state of the system.
    """

    def __init__(self, t0=0):
        self.t0 = t0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_S = 0

        self.contributions = []
        self.contributions_map = {}
        self.ncontr = 0

        self.origin = Frame()

        self.origin.name = "origin"
        self.add(self.origin)

    def add(self, *contrs):
        """Adds contributions to the system.

        Parameters
        ----------
        contrs : object or list
            Single object or list of objects to add to the system.
        """
        for contr in contrs:
            if not contr in self.contributions:
                self.contributions.append(contr)
                self.ncontr += 1

    def assemble(self):
        """Assembles the system, i.e., counts degrees of freedom, sets connectivities and assembles global initial state."""
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_S = 0
        q0 = []
        u0 = []

        for p in properties:
            setattr(self, f"_{self.__class__.__name__}__{p}_contr", [])

        for contr in self.contributions:
            contr.t0 = self.t0
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if hasattr(contr, p) and callable(getattr(contr, p)):
                    getattr(self, f"_{self.__class__.__name__}__{p}_contr").append(
                        contr
                    )

            # if contribution has position degrees of freedom address position coordinates
            if hasattr(contr, "nq"):
                contr.my_qDOF = np.arange(0, contr.nq) + self.nq
                contr.qDOF = contr.my_qDOF.copy()
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, "nu"):
                contr.my_uDOF = np.arange(0, contr.nu) + self.nu
                contr.uDOF = contr.my_uDOF.copy()
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())

            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, "nla_g"):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g

            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, "nla_S"):
                contr.la_SDOF = np.arange(0, contr.nla_S) + self.nla_S
                self.nla_S += contr.nla_S

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

        #  initial conditions
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)
        self.la_g0 = np.zeros(self.nla_g)

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    def step_callback(self, t, q):
        pass

    #####################
    # equations of motion
    #####################
    def h(self, t):
        h = np.zeros(self.nu, dtype=np.float64)
        for contr in self.__h_contr:
            h[contr.uDOF] += contr.h(t)
        return h

    def h_q(self, t):
        h_q = np.zeros((self.nu, self.nq))
        for contr in self.__h_q_contr:
            h_q[np.ix_(contr.uDOF, contr.qDOF)] += contr.h_q(t)
        return h_q

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t):
        g = np.zeros(self.nla_g, dtype=np.float64)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t)
        return g

    def g_q(self, t):
        g_q = np.zeros((self.nla_g, self.nq))
        for contr in self.__g_contr:
            g_q[np.ix_(contr.la_gDOF, contr.qDOF)] += contr.g_q(t)
        return g_q

    def W_g(self, t):
        W_g = np.zeros((self.nu, self.nla_g))
        for contr in self.__g_contr:
            W_g[np.ix_(contr.uDOF, contr.la_gDOF)] += contr.W_g(t)
        return W_g

    def Wla_g_q(self, t):
        Wla_g_q = np.zeros((self.nu, self.nq))
        for contr in self.__g_contr:
            Wla_g_q[np.ix_(contr.uDOF, contr.qDOF)] += contr.Wla_g_q(t)
        return Wla_g_q

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self):
        g_S = np.zeros(self.nla_S, dtype=np.float64)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S()
        return g_S

    def g_S_q(self):
        g_S_q = np.zeros((self.nla_S, self.nq))
        for contr in self.__g_S_contr:
            g_S_q[np.ix_(contr.la_SDOF, contr.qDOF)] += contr.g_S_q()
        return g_S_q

    def connect_state(self, q, la_g):
        for contr in self.contributions:
            if hasattr(contr, "qDOF") and len(contr.qDOF) > 0:
                contr.q = q[contr.qDOF[0] : contr.qDOF[-1] + 1]
                assert np.shares_memory(contr.q, q)

            if hasattr(contr, "la_gDOF") and len(contr.la_gDOF) > 0:
                contr.la_g = la_g[contr.la_gDOF[0] : contr.la_gDOF[-1] + 1]
                assert np.shares_memory(contr.la_g, la_g)
