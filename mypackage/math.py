import numpy as np

angle_singular = 0.0

eye3 = np.eye(3, dtype=float)


def Spurrier(R: np.ndarray) -> np.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = np.zeros(4, dtype=float)
    decision[:3] = np.diag(R)
    decision[3] = np.trace(R)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=float)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * R[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (R[k, j] - R[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (R[j, i] + R[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (R[k, i] + R[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (R[2, 1] - R[1, 2]) / (4 * quat[0])
        quat[2] = (R[0, 2] - R[2, 0]) / (4 * quat[0])
        quat[3] = (R[1, 0] - R[0, 1]) / (4 * quat[0])

    return quat

Log_SO3_quat = Spurrier


def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        # Nuetzi2016 (3.31) and Rucker2018 (13)
        P2 = P @ P
        return eye3 + (2 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p))
    else:
        # returns always an orthogonal matrix, but not necessary normalized,
        # see Egeland2002 (6.163)
        return (p0**2 - p @ p) * eye3 + np.outer(p, 2 * p) + 2 * p0 * ax2skew(p)


def Exp_SO3_quat_p(P, normalize=True):
    """Derivative of Exp_SO3_quat with respect to P."""
    p0, p = P[0, None], P[1:]
    p_tilde = ax2skew(p)
    p_tilde_p = ax2skew_a()

    if normalize:
        P2 = P @ P
        A_P = np.einsum(
            "ij,k->ijk", p0 * p_tilde + ax2skew_squared(p), -(4 / (P2 * P2)) * P
        )
        s2 = 2 / P2
        A_P[:, :, 0] += s2 * p_tilde
        A_P[:, :, 1:] += (
            s2 * p0 * p_tilde_p
            + np.einsum("ijl,jk->ikl", p_tilde_p, s2 * p_tilde)
            + np.einsum("ij,jkl->ikl", s2 * p_tilde, p_tilde_p)
        )
    else:
        A_P = np.zeros((3, 3, 4), dtype=P.dtype)
        A_P[:, :, 0] = 2 * p0 * eye3 + 2 * ax2skew(p)
        A_P[:, :, 1:] = -np.multiply.outer(eye3, 2 * p) + 2 * p0 * ax2skew_a()
        A_P[0, :, 1:] += 2 * p[0] * eye3
        A_P[1, :, 1:] += 2 * p[1] * eye3
        A_P[2, :, 1:] += 2 * p[2] * eye3
        A_P[0, :, 1] += 2 * p
        A_P[1, :, 2] += 2 * p
        A_P[2, :, 3] += 2 * p

    return A_P


def T_SO3_quat(P, normalize=True):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        return (2 / (P @ P)) * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    else:
        return 2 * (P @ P) * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))


def T_SO3_inv_quat(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0, None], P[1:]
    if normalize:
        return 0.5 * np.vstack((-p.T, p0 * eye3 + ax2skew(p)))
    else:
        return 1 / (2 * (P @ P) ** 2) * np.vstack((-p.T, p0 * eye3 + ax2skew(p)))


def T_SO3_quat_P(P, normalize=True):
    p0, p = P[0, None], P[1:]
    P2 = P @ P
    matrix = np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    if normalize:
        factor = 2 / P2
        factor_P = -4 * P / P2**2
    else:
        factor = 2 * P2
        factor_P = 4 * P

    T_P = np.multiply.outer(matrix, factor_P)
    T_P[:, 0, 1:] -= factor * eye3
    T_P[:, 1:, 0] += factor * eye3
    T_P[:, 1:, 1:] -= factor * ax2skew_a()

    return T_P



def Exp_SO3(psi: np.ndarray) -> np.ndarray:
    """SO(3) exponential function, see Crisfield1999 above (4.1) and 
    Park2005 (12).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Park2005: https://doi.org/10.1109/TRO.2005.852253
    """
    angle = norm(psi)
    if angle > angle_singular:
        # Park2005 (12)
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        beta2 = (1.0 - ca) / (angle * angle)
        return eye3 + alpha * ax2skew(psi) + beta2 * ax2skew_squared(psi)
    else:
        # first order approximation
        return eye3 + ax2skew(psi)


def Log_SO3(A: np.ndarray) -> np.ndarray:
    ca = 0.5 * (np.trace(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    # fmt: off
    psi = 0.5 * np.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ], dtype=A.dtype)
    # fmt: on

    if angle > angle_singular and angle < np.pi:
        psi *= angle / np.sqrt(1.0 - ca * ca)
    return psi


def norm(a: np.ndarray) -> float:
    """Euclidean norm of an array of arbitrary length."""
    return np.sqrt(a @ a)


def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on
    
def ax2skew_a():
    """
    Partial derivative of the `ax2skew` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    A = np.zeros((3, 3, 3), dtype=float)
    A[1, 2, 0] = -1
    A[2, 1, 0] = 1
    A[0, 2, 1] = 1
    A[2, 0, 1] = -1
    A[0, 1, 2] = -1
    A[1, 0, 2] = 1
    return A

def ax2skew_squared(a: np.ndarray) -> np.ndarray:
    """Computes the product of a skew-symmetric matrix with itself from a given axial vector."""
    assert a.size == 3
    a1, a2, a3 = a
    # fmt: off
    return np.array([
        [-a2**2 - a3**2,              a1 * a2,              a1 * a3],
        [             a2 * a1, -a1**2 - a3**2,              a2 * a3],
        [             a3 * a1,              a3 * a2, -a1**2 - a2**2],
    ], dtype=a.dtype)
    # fmt: on


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector product of two 3D vectors."""
    assert a.size == 3
    assert b.size == 3
    # fmt: off
    return np.array([a[1] * b[2] - a[2] * b[1], \
                     a[2] * b[0] - a[0] * b[2], \
                     a[0] * b[1] - a[1] * b[0] ])
    # fmt: on


def interp1d(x, y, xi):
    """
    linear interpolation, support multidimensional y
    """

    idx = np.searchsorted(x, xi, side="right") - 1
    if idx == len(y) - 1:
        return y[-1]
    else:
        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        t = (xi - x0) / (x1 - x0)
        yi = y0 + (y1 - y0) * t
        return yi
    
def lagrange_basis_with_derivative(xs, i):
    xs = np.array(xs, dtype=float)
    xi = xs[i]

    denom = 1.0
    for j, xj in enumerate(xs):
        if j != i:
            denom *= (xi - xj)

    def L(x):
        num = 1.0
        for j, xj in enumerate(xs):
            if j != i:
                num *= (x - xj)
        return num / denom

    def dL(x):
        res = 0.0

        for k, xk in enumerate(xs):
            if k != i:
                num = 1.0
                for j, xj in enumerate(xs):
                    if j != i and j != k:
                        num *= (x - xj)
                res += num 

        return res / denom

    return L, dL