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
                if not hasattr(contr, "name"):
                    contr.name = "contr" + str(self.ncontr)

                if contr.name in self.contributions_map:
                    new_name = contr.name + "_contr" + str(self.ncontr)
                    print(
                        f"There is another contribution named '{contr.name}' which is already part of the system. Changed the name to '{new_name}' and added it to the system."
                    )
                    contr.name = new_name
                self.contributions_map[contr.name] = contr
                self.ncontr += 1
            else:
                raise ValueError(f"contribution {str(contr)} already added")

    def get_contribution_list(self, contr):
        return getattr(self, f"_{self.__class__.__name__}__{contr}_contr")

    def assemble(self):
        """Assembles the system, i.e., counts degrees of freedom, sets connectivities and assembles global initial state."""
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_c = 0
        self.nla_tau = 0
        self.ntau = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0
        q0 = []
        u0 = []
        self.constant_force_reservoir = False

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

        # compute consisten initial conditions
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)

        # compute consistent initial conditions
        # normalize quaternions etc.
        q0, u0 = self.step_callback(self.t0, self.q0, self.u0)
        self.la_g0 = np.zeros(self.nla_g)

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    def step_callback(self, t, q, u):
        for contr in self.__step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.step_callback(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    #####################
    # equations of motion
    #####################
    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for contr in self.__h_contr:
            h[contr.uDOF] += contr.h(t, q[contr.qDOF], u[contr.uDOF])
        return h

    def h_q(self, t, q, u):
        h_q = np.zeros((self.nu, self.nq))
        for contr in self.__h_q_contr:
            h_q[np.ix_(contr.uDOF, contr.qDOF)] += contr.h_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return h_q

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_q(self, t, q):
        g_q = np.zeros((self.nla_g, self.nq))
        for contr in self.__g_contr:
            g_q[np.ix_(contr.la_gDOF, contr.qDOF)] += contr.g_q(t, q[contr.qDOF])
        return g_q

    def W_g(self, t, q):
        W_g = np.zeros((self.nu, self.nla_g))
        for contr in self.__g_contr:
            W_g[np.ix_(contr.uDOF, contr.la_gDOF)] += contr.W_g(t, q[contr.qDOF])
        return W_g

    def Wla_g_q(self, t, q, la_g):
        Wla_g_q = np.zeros((self.nu, self.nq))
        for contr in self.__g_contr:
            Wla_g_q[np.ix_(contr.uDOF, contr.qDOF)] += contr.Wla_g_q(
                t, q[contr.qDOF], la_g[contr.la_gDOF]
            )
        return Wla_g_q

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S(t, q[contr.qDOF])
        return g_S

    def g_S_q(self, t, q):
        g_S_q = np.zeros((self.nla_S, self.nq))
        for contr in self.__g_S_contr:
            g_S_q[np.ix_(contr.la_SDOF, contr.qDOF)] += contr.g_S_q(t, q[contr.qDOF])
        return g_S_q
