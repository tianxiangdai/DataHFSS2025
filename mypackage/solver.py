import numpy as np
from scipy.optimize import OptimizeResult
from scipy.linalg import solve as spsolve

from tqdm import tqdm
from warnings import warn
from collections import namedtuple


class Solution:
    """Class to store solver outputs."""

    def __init__(
        self,
        system,
        t,
        q,
        u=None,
        u_dot=None,
        la_g=None,
        la_gamma=None,
        la_c=None,
        la_N=None,
        la_F=None,
        **kwargs,
    ):
        self.system = system
        self.t = t
        self.q = q
        self.u = u
        self.u_dot = u_dot
        self.la_g = la_g
        self.la_gamma = la_gamma
        self.la_c = la_c
        self.la_N = la_N
        self.la_F = la_F
        self.solver_summary = None

        self.__dict__.update(**kwargs)

    def __iter__(self):
        return self.SolutionIterator(self)

    class SolutionIterator:
        def __init__(self, solution) -> None:
            self._solution = solution
            self.keys = [*self._solution.__dict__.keys()]
            # remove non-iterable keys
            self.keys.remove("solver_summary")
            self.keys.remove("system")

            self._index = 0
            self._retVal = namedtuple("Result", self.keys)

        def __next__(self):
            if self._index < len(self._solution.t):
                try:
                    result = self._retVal(
                        *(
                            (
                                None
                                if self._solution.__getattribute__(key) is None
                                else (
                                    self._solution.__getattribute__(key)[:, self._index]
                                    if self._solution.__getattribute__(key).shape[0]
                                    == 0
                                    else self._solution.__getattribute__(key)[
                                        self._index
                                    ]
                                )
                            )
                            for key in self.keys
                        )
                    )
                except:
                    RuntimeWarning("Solution iterator failed.")
                self._index += 1
                return result
            raise StopIteration


def fsolve(
    fun, x0, jac, fun_args=(), jac_args=(), max_iter=20, atol=1e-6, rtol=1e-6
) -> tuple[np.ndarray, bool, float, int, np.ndarray]:
    """Solve a nonlinear system of equations using (inexact) Newton method.
    This function is inspired by scipy's `solve_collocation_system` found
    in `scipy.integrate._ivp.radau`. Absolute and relative errors are used 
    to terminate the iteration in accordance with Kelly1995 (1.12). See also 
    Hairer1996 below (8.21).

    Parameters
    ----------
    fun : callable
        Nonlinear function with signature `fun(x, *fun_args)`.
    x0 : ndarray, shape (n,)
        Initial guess.
    jac : callable, SuperLU, optional
        Function defining the sparse Jacobian of `fun`. Alternatvely, this
        can be an `SuperLU` object. Then, an inexact Newton method is
        performed, see `inexact`.
    fun_args: tuple
        Additional arguments passed to `fun`.
    jac_args: tuple
        Additional arguments passed to `jac`.
    inexact: Bool, optional
        Apply inexact Newton method (Newton chord) with constant `J = jac(x0)`.
    options: SolverOptions
        Defines all required solver options.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: `x` the solution array, `success` a
        Boolean flag indicating if the optimizer exited successfully, `error` 
        the relative error, `fun` the current function value and `nit`, 
        `nfev`, `njev` the counters for number of iterations, function and 
        Jacobian evaluations, respectively.

    References
    ----------
    Kelly1995: https://epubs.siam.org/doi/book/10.1137/1.9780898718898 \\
    Haireri1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """
    nit = 0
    nfev = 0
    njev = 0

    if not isinstance(fun_args, tuple):
        fun_args = (fun_args,)
    if not jac_args:
        jac_args = fun_args
    elif not isinstance(jac_args, tuple):
        jac_args = (jac_args,)

    # wrap function
    def fun(x, *args, f=fun):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(f(x, *args))

    # wrap jacobian
    def jacobian(x, *args):
        nonlocal njev
        njev += 1
        return jac(x, *args)

    def solve(x, rhs):
        return spsolve(jacobian(x, *jac_args), rhs)

    # eliminate round-off errors
    Delta_x = np.zeros_like(x0)
    x = x0 + Delta_x

    # initial function value
    f = np.atleast_1d(fun(x, *fun_args))

    # scaling with relative and absolute tolerances
    scale = atol + np.abs(f) * rtol

    # error of initial guess
    error = np.linalg.norm(f / scale) / scale.size**0.5
    converged = error < 1

    # Newton loop
    if not converged:
        for i in range(max_iter):
            # Newton update
            dx = solve(x, f)
            Delta_x -= dx
            x = x0 + Delta_x

            # new function value, error and convergence check
            f = np.atleast_1d(fun(x, *fun_args))
            error = np.linalg.norm(f / scale) / scale.size**0.5
            converged = error < 1
            if converged:
                break

        if not converged:
            warn(f"fsolve is not converged after {i} iterations with error {error:.2e}")

        nit = i + 1

    return OptimizeResult(
        x=x,
        success=converged,
        error=error,
        fun=f,
        nit=nit,
        nfev=nfev,
        njev=njev,
    )


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
        self, system, n_load_steps=1, verbose=True, max_iter=20, atol=1e-6, rtol=1e-6
    ):
        self.system = system
        # self.options = options
        self.verbose = verbose
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = len(self.load_steps)

        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol

        self.len_t = len(str(self.nt))
        self.len_maxIter = len(str(max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_S],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq],
                dtype=int,
            )
        )

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0))
        nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        # memory allocation
        self.x = np.zeros((self.nt, nx), dtype=float)
        self.x[0] = x0
        nf = system.nu + system.nla_g + system.nla_S

        # allocate memory
        self.__x = x0.copy()
        self.__q = self.__x[: self.split_x[0]]
        self.__la_g = self.__x[self.split_x[0] :]
        self.system.connect_state(self.__q, self.__la_g)
        self.__F = np.zeros(nf, dtype=float)
        self.__h = self.__F[: self.split_f[0]]
        self.__g = self.__F[self.split_f[0] : self.split_f[1]]
        self.__g_S = self.__F[self.split_f[1] :]
        self.__J = np.zeros((nf, nx), dtype=float)
        self.__K = self.__J[: self.split_f[0], : self.split_x[0]]
        self.__W_g = self.__J[: self.split_f[0], self.split_x[0] :]
        self.__g_q = self.__J[self.split_f[0] : self.split_f[1], : self.split_x[0]]
        self.__g_S_q = self.__J[self.split_f[1] :, : self.split_x[0]]

    def fun(self, x, t):
        # unpack unknowns
        self.__x[:] = x
        la_g = x[self.split_x[0] :]
        # evaluate quantites that are required for computing the residual and
        # the jacobian
        self.__W_g[:] = self.system.W_g(t)

        # static equilibrium
        self.__h[:] = self.system.h(t) + self.__W_g @ la_g
        self.__g[:] = self.system.g(t)
        self.__g_S[:] = self.system.g_S()
        return self.__F

    def jac(self, x, t):
        # evaluate additionally required quantites for computing the jacobian
        self.__K[:] = self.system.h_q(t) + self.system.Wla_g_q(t)
        self.__g_q[:] = self.system.g_q(t)
        self.__g_S_q[:] = self.system.g_S_q()
        return self.__J

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.max_iter};"
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
                max_iter=self.max_iter,
                atol=self.atol,
                rtol=self.rtol,
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
                )

            # solver step callback
            self.system.step_callback(self.load_steps[i], self.x[i, : self.split_x[0]])

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
            la_g=self.x[: i + 1, self.split_x[0] :],
        )


class StaticSolver(Newton):
    def __init__(
        self, system, n_load_steps=1, verbose=True, max_iter=20, atol=1e-6, rtol=1e-6
    ):
        super().__init__(
            system, n_load_steps, verbose, max_iter=max_iter, atol=atol, rtol=rtol
        )
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
        self.x0 = np.concatenate((system.q0, system.la_g0))

    def solve(self, warm_start=True):
        if warm_start and self.x0 is not None:
            self.x[0] = self.x0
        res = super().solve()
        if warm_start:
            self.x0 = self.x[-1]
        return res
