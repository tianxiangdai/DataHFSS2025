import numpy as np


class Force_line_distributed:
    def __init__(self, force, rod):
        r"""Line distributed dead load for rods

        Parameters
        ----------
        force : np.ndarray (3,)
            Force w.r.t. inertial I-basis as a callable function in time t and
            rod position xi.
        rod : CosseratRod

        """
        if not callable(force):
            self.force = lambda t, xi: force
        else:
            self.force = force
        self.rod = rod

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

    #####################
    # equations of motion
    #####################
    def h(self, t, q, u):
        h = np.zeros(self.rod.nu, dtype=np.float64)

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[i]
            qwi = self.rod.qw[i]
            Ji = self.rod.J[i]

            # compute local force vector
            h_qp = self.force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.rod.nnodes):
                h[self.rod.nodalDOF_r[node]] += (
                    self.rod.N[i, node] * h_qp
                )
        return h

    def h_q(self, t, q, u):
        return None
