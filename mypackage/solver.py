import numpy as np
from scipy.sparse import bmat
from tqdm import tqdm

from cardillo.math.fsolve import fsolve
from cardillo.solver.solver_options import SolverOptions
from cardillo.solver.solution import Solution

class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. Forces and
    bilateral constraint functions are incremented in each load step if they
    depend on the time t in [0, 1]. Thus, a force controlled Newton-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing constant forces and time dependent constraint functions.
    """

    def __init__(
        self,
        system,
        n_load_steps=1,
        verbose=True,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.verbose = verbose
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = len(self.load_steps)

        self.len_t = len(str(self.nt))
        self.len_maxIter = len(str(self.options.newton_max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu
        self.nla_N = system.nla_N

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_c, system.nla_S],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq, system.nla_g, system.nla_c],
                dtype=int,
            )
        )

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0, system.la_c0, system.la_N0))
        nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        # memory allocation
        self.x = np.zeros((self.nt, nx), dtype=float)
        self.x[0] = x0

    def fun(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        # csr is used for efficient matrix vector multiplication, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        self.W_g = self.system.W_g(t, q, format="csr")

        # static equilibrium
        F = np.zeros_like(x)
        F[: self.split_f[0]] = (
            self.system.h(t, q, self.u0)
            + self.W_g @ la_g
        )
        F[self.split_f[0] : self.split_f[1]] = self.system.g(t, q)
        F[self.split_f[2] : self.split_f[3]] = self.system.g_S(t, q)
        return F

    def jac(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        K = (
            self.system.h_q(t, q, self.u0)
            + self.system.Wla_g_q(t, q, la_g)
        )
        g_q = self.system.g_q(t, q)
        g_S_q = self.system.g_S_q(t, q)

        # fmt: off
        return bmat([[      K, self.W_g], 
                     [    g_q,     None],
                     [  g_S_q,     None]], format="csc")
        # fmt: on

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.options.newton_max_iter};"
            f" error {error:.4e}"
        )

    def solve(self):
        pbar = range(0, self.nt)
        if self.verbose:
            pbar = tqdm(pbar, leave=True)
        for i in pbar:
            sol = fsolve(
                self.fun,
                self.x[i],
                jac=self.jac,
                fun_args=(self.load_steps[i],),
                jac_args=(self.load_steps[i],),
                options=self.options,
            )
            self.x[i] = sol.x
            if self.verbose:
                pbar.set_description(self.__pbar_text(i, sol.nit, sol.error))

            if not sol.success and not self.options.continue_with_unconverged:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{self.len_t}d}/{self.nt}"
                )
                return Solution(
                    system=self.system,
                    t=self.load_steps[: i + 1],
                    q=self.x[: i + 1, : self.split_x[0]],
                    u=np.zeros((i + 1, self.nu)),
                    la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
                    la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
                    la_N=self.x[: i + 1, self.split_x[2] :],
                )

            # solver step callback
            self.x[i, : self.split_x[0]], _ = self.system.step_callback(
                self.load_steps[i], self.x[i, : self.split_x[0]], self.u0
            )

            # warm start for next step; store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()
        return Solution(
            self.system,
            t=self.load_steps,
            q=self.x[: i + 1, : self.split_x[0]],
            u=np.zeros((len(self.load_steps), self.nu)),
            la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
            la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
            la_N=self.x[: i + 1, self.split_x[2] :],
        )

class StaticSolver(Newton):
    def __init__(self, system, n_load_steps=1, verbose=True, options=SolverOptions()):
        super().__init__(system, n_load_steps, verbose, options)
        self.n_load_steps = self.nt - 1
        self.x0 = None

    def set_load_steps(self, n_load_steps):
        if n_load_steps == self.n_load_steps:
            return
        self.n_load_steps = n_load_steps
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = n_load_steps + 1
        self.len_t = len(str(self.nt))
        x0 = self.x[0]
        self.x = np.zeros((self.nt, len(self.x[0])), dtype=np.float64)
        self.x[0] = x0

    def renew_initial_state(self):
        system = self.system
        self.x0 = np.concatenate((system.q0, system.la_g0, system.la_c0, system.la_N0))

    def solve(self, warm_start=True):
        if warm_start and self.x0 is not None:
            self.x[0] = self.x0
        res = super().solve()
        if warm_start:
            self.x0 = self.x[-1]
        return res
