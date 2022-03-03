import numpy as np
import numba as nb
from numba import int64, float32, float64, int32

@nb.jit(nopython=True)
def _squeeze_output(out):
    """
    Remove single-dimensional entries from array and convert to scalar,
    if necessary.

    """
    out = out.squeeze()
    if out.ndim == 0:
        out = out[()]
    return out

@nb.jit(nopython=True)
def normal_logpdf(x, mu, cov):
    """
    Multivariate normal logpdf, numpy native implementation
    :param x:
    :param mu:
    :param cov:
    :return:
    """
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(np.log(part1) + part2)


@nb.jit(nopython=True)
def invwishartrand_prec(nu, phi):
    return np.linalg.inv(wishartrand(nu, phi))


@nb.jit(nopython=True)
def invwishartrand(nu, phi):
    return np.linalg.inv(wishartrand(nu, np.linalg.inv(phi)))


@nb.jit(nopython=True)
def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = np.linalg.cholesky(phi)
    foo = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                foo[i, j] = np.sqrt(np.random.chisquare(nu - (i + 1) + 1))
            else:
                foo[i, j] = np.random.normal(0, 1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


#@nb.jit(fastmath=True,error_model='numpy')
@nb.vectorize([float64(float64), float64(float64), float64(int32), float32(int64)])
def gammaln(z):
    """Numerical Recipes 6.1"""
    coefs = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5])

    y = z#.copy()
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y += 1.0
        ser = ser + coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out


@nb.jit(nopython=True)
def t_logpdf(x, loc, cov, df):
    """
    Utility method `pdf`, `logpdf` for parameters.
    """
    dim = len(x)
    dev = x - loc
    maha = ((dev).T.dot(np.linalg.inv(cov))).dot((dev))

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * np.log(np.linalg.det(cov))
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E
