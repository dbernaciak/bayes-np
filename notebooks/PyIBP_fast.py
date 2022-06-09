"""
PyIBP_fast

Implements fast Gibbs sampling for the linear-Gaussian
infinite latent feature model (IBP).

Adapted from 2009 David Andrzejewski (andrzeje@cs.wisc.edu)

This version introduces JITed functions which speed the code up significantly.
"""

import numpy as np
import numpy.random as nr
import scipy.stats as st
import numba as nb
from numba import int64, float32, float64, int32
import pdb

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')


class PyIBP(object):
    """
    Implements fast Gibbs sampling for the linear-Gaussian
    infinite latent feature model (IBP)
    """

    #
    # Initialization methods
    #

    def __init__(self, data, alpha, sigma_x, sigma_a,
                 missing=None, useV=False, initZV=None):
        """ 
        data = NxD NumPy data matrix (should be centered)

        alpha = Fixed IBP hyperparam for OR (init,a,b) tuple where
        (a,b) are Gamma hyperprior shape and rate/inverse scale
        sigma_x = Fixed noise std OR (init,a,b) tuple (same as alpha)
        sigma_a = Fixed weight std OR (init,a,b) tuple (same as alpha)

        OPTIONAL ARGS
        missing = boolean/binary 'missing data' mask (1=missing entry)
        useV = Are we using real-valued latent features? (default binary)
        initZV = Optional initial state for the latent         
        """
        # Data matrix
        self.X = data
        (self.N, self.D) = data.shape
        # IBP hyperparameter
        if (type(alpha) == tuple):
            (self.alpha, self.alpha_a, self.alpha_b) = alpha
        else:
            (self.alpha, self.alpha_a, self.alpha_b) = (alpha, None, None)
        # Noise variance hyperparameter
        if (type(sigma_x) == tuple):
            (self.sigma_x, self.sigma_xa, self.sigma_xb) = sigma_x
        else:
            (self.sigma_x, self.sigma_xa, self.sigma_xb) = (sigma_x, None, None)
        # Weight variance hyperparameter
        if (type(sigma_a) == tuple):
            (self.sigma_a, self.sigma_aa, self.sigma_ab) = sigma_a
        else:
            (self.sigma_a, self.sigma_aa, self.sigma_ab) = (sigma_a, None, None)
        # Are we using weighted latent features?
        self.useV = useV
        # Do we have user-supplied initial latent feature values?
        if (initZV == None):
            # Initialze Z from IBP(alpha)
            self.initZ()
            # Initialize V from N(0,1) if necessary
            if (self.useV):
                self.initV()
        else:
            self.ZV = initZV
            self.K = self.ZV.shape[1]
            self.m = (self.ZV != 0).astype(int).sum(axis=0)
        # Sample missing data entries if necessary
        self.missing = missing
        if (missing != None):
            self.sample_X()

    def initV(self):
        """ Init latent feature weights V accoring to N(0,1) """
        for (i, k) in zip(*self.ZV.nonzero()):
            self.ZV[i, k] = nr.normal(0, 1)

    def initZ(self):
        """ Init latent features Z according to IBP(alpha) """
        Z = np.ones((0, 0))
        for i in range(1, self.N + 1):  # generate IBP
            # Sample existing features
            zi = (nr.uniform(0, 1, (1, Z.shape[1])) <
                  (Z.sum(axis=0).astype(np.float) / i))
            # Sample new features
            knew = st.poisson.rvs(self.alpha / i)
            zi = np.hstack((zi, np.ones((1, knew))))
            # Add to Z matrix
            Z = np.hstack((Z, np.zeros((Z.shape[0], knew))))
            Z = np.vstack((Z, zi))
        self.ZV = Z
        self.K = self.ZV.shape[1]
        # Calculate initial feature counts
        self.m = (self.ZV != 0).astype(int).sum(axis=0)

    #
    # Convenient external methods
    #

    def fullSample(self):
        """ Do all applicable samples """
        self._sample_Z()
        if self.missing is not None:
            self.sample_X()
        if self.alpha_a is not None:
            self.alpha = sample_alpha(self.alpha_a, self.alpha_b, self.N, self.m)
            # print(self.alpha)
        if self.sigma_xa is not None:
            self.sampleSigma()

    def logLike(self):
        """
        Calculate log-likelihood P(X,Z)
        (or P(X,Z,V) if applicable)
        """
        liketerm = self.logPX(calc_M(self.ZV, self.K, self.sigma_x, self.sigma_a), self.ZV)
        ibpterm = self.logIBP()
        if (self.useV):
            vterm = self.logPV()
            return liketerm + ibpterm + vterm
        else:
            return liketerm + ibpterm

    def weights(self):
        """ Return E[A|X,Z] """
        return self.postA(self.X, self.ZV)[0]

    #
    # Actual sampling methods
    #

    def sampleSigma(self):
        """ Sample feature/noise variances """
        # Posterior over feature weights A
        (mean_A, covarA) = self.postA(self.X, self.ZV)
        # sigma_x
        vars = np.dot(self.ZV, np.dot(covarA, self.ZV.T)).diagonal()
        var_x = (np.power(self.X - np.dot(self.ZV, mean_A), 2)).sum()
        var_x += self.D * vars.sum()
        n = float(self.N * self.D)
        post_shape = self.sigma_xa + n / 2
        post_scale = float(1) / (self.sigma_xb + var_x / 2)
        tau_x = nr.gamma(post_shape, scale=post_scale)
        self.sigma_x = np.sqrt(float(1) / tau_x)
        # sigma_a
        var_a = covarA.trace() * self.D + np.power(mean_A, 2).sum()
        n = float(self.K * self.D)
        post_shape = self.sigma_aa + n / 2
        post_scale = float(1) / (self.sigma_ab + var_a / 2)
        tau_a = st.gamma.rvs(post_shape, scale=post_scale)
        self.sigma_a = np.sqrt(float(1) / tau_a)

    def sample_alpha(self):
        """ Sample alpha from conjugate posterior """
        post_shape = self.alpha_a + self.m.sum()
        post_scale = float(1) / (self.alpha_b + self.N)
        self.alpha = nr.gamma(post_shape, scale=post_scale)

    def sample_X(self):
        """ Take single sample missing data entries in X """
        # Calculate posterior mean/covar --> info
        (mean_A, covarA) = self.postA(self.X, self.ZV)
        (infoA, hA) = to_info(mean_A, covarA)
        # Find missing observations
        xis = np.nonzero(self.missing.max(axis=1))[0]
        for i in xis:
            # Get (z,x) for this data point
            (zi, xi) = (np.reshape(self.ZV[i, :], (1, self.K)),
                        np.reshape(self.X[i, :], (1, self.D)))
            # Remove this observation
            infoA_i = update_info(infoA, zi, -1, self.sigma_x)
            hA_i = update_h(hA, zi, xi, -1, self.sigma_x)
            # Convert back to mean/covar
            mean_A_i, covarA_i = fromInfo(infoA_i, hA_i)
            # Resample xi
            meanXi, covarXi = like_xi(zi, mean_A_i, covarA_i, self.sigma_x)
            newxi = nr.normal(meanXi, np.sqrt(covarXi))
            # Replace missing features
            ks = np.nonzero(self.missing[i, :])[0]
            self.X[i, ks] = newxi[0][ks]

    def _sample_Z(self):
        self.ZV, self.K, self.m = sample_Z(self.N, self.X, self.ZV, self.K, self.D, self.m, self.alpha, self.sigma_x,
                                           self.sigma_a, self.useV)

    def sample_report(self, sampleidx):
        """ Print IBP sample status """
        return {
            "iter": sampleidx,
            "collapsed_loglike": self.logLike(),
            "K": self.K,
            "alpha": self.alpha,
            "sigma_x": self.sigma_x,
            "sigma_a": self.sigma_a
        }

    def weightReport(self, trueWeights=None, round=False):
        """ Print learned weights (vs ground truth if available) """
        if (trueWeights != None):
            print('\nTrue weights (A)')
            print(str(trueWeights))
        print('\nLearned weights (A)')
        # Print rounded or actual weights?
        if (round):
            print(str(self.weights().astype(int)))
        else:
            print(np.array_str(self.weights(), precision=2, suppress_small=True))
        print('')
        # Print V matrix if applicable
        if (self.useV):
            print('\nLatent feature weights (V)')
            print(np.array_str(self.ZV, precision=2))
            print('')
        # Print 'popularity' of latent features
        print('\nLatent feature counts (m)')
        print(np.array_str(self.m))

    #
    # Bookkeeping and calculation methods
    #

    def logPV(self):
        """ Log-likelihood of real-valued latent features V """
        return _logPV(self.ZV)

    # to be migrated
    def logIBP(self):
        """ Calculate IBP prior contribution log P(Z|alpha) """
        return _logIBP(self.ZV, self.alpha, self.m)

    # to be migrated
    def postA(self, X, Z):
        """ Mean/covar of posterior over weights A """
        return _postA(X, Z, self.K, self.sigma_x, self.sigma_a)

    # to be migrated
    def logPX(self, M, Z):
        return PyIBP._logPX(M, Z, self.N, self.D, self.K, self.X, self.sigma_x, self.sigma_a)

    #
    # Pure functions (these don't use state or additional params)
    #

    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def _logPX(M, Z, N, D, K, X, sigma_x, sigma_a):
        """ Calculate collapsed log likelihood of data"""
        lp = -0.5 * N * D * np.log(2 * np.pi)
        lp -= (N - K) * D * np.log(sigma_x)
        lp -= K * D * np.log(sigma_a)
        lp -= 0.5 * D * np.log(np.linalg.det(np.linalg.inv(M)))
        iminzmz = np.eye(N) - np.dot(Z, np.dot(M, Z.T))
        lp -= (0.5 / (sigma_x ** 2)) * np.trace(
            np.dot(X.T, np.dot(iminzmz, X)))
        return lp

    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def logFact(n):
        return gammaln(n + 1)

    @staticmethod
    def centerData(data):
        return data - PyIBP.featMeans(data)

    @staticmethod
    def featMeans(data, missing=None):
        """ Replace all columns (features) with their means """
        (N, D) = data.shape
        if (missing == None):
            return np.tile(data.mean(axis=0), (N, 1))
        else:
            # Sanity check on 'missing' mask
            # (ensure no totally missing data or features)
            assert (all(missing.sum(axis=0) < N) and
                    all(missing.sum(axis=1) < D))
            # Calculate column means without using the missing data
            censored = data * (np.ones((N, D)) - missing)
            censoredmeans = censored.sum(axis=0) / (N - missing.sum(axis=0))
            return np.tile(censoredmeans, (N, 1))


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

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y += 1.0
        ser = ser + coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out


@nb.jit(nopython=True, fastmath=True)
def sample_alpha(alpha_a, alpha_b, N, m):
    """ Sample alpha from conjugate posterior """
    post_shape = alpha_a + m.sum()
    post_scale = float(1) / (alpha_b + N)
    return nr.gamma(post_shape, scale=post_scale)


@nb.jit(["float64(float64)"], nopython=True, fastmath=True)
def logUnif(v):
    """
    Sample uniformly from [0, exp(v)] in the log-domain
    (derive via transform f(x)=log(x) and some calculus...)
    """
    return v + np.log(nr.uniform(0, 1))


@nb.jit(["float64[:, :](float64[:, :], int64, float64, float64)"], nopython=True, fastmath=True)
def calc_M(Z, K, sigma_x, sigma_a):
    """ Calculate M = (Z' * Z - (sigmax^2) / (sigmaa^2) * I)^-1 """
    return np.linalg.inv(np.dot(Z.T, Z) + (sigma_x ** 2) / (sigma_a ** 2) * np.eye(K))


@nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :], nb.int64, nb.float64, nb.float64),
        nopython=True, fastmath=True)
def _postA(X, Z, K, sigma_x, sigma_a):
    M = calc_M(Z, K, sigma_x, sigma_a)
    mean_A = np.dot(M, np.dot(Z.T, X))
    covarA = sigma_x ** 2 * M
    return mean_A, covarA


@nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :]), nopython=True, fastmath=True)
def to_info(mean_A, covarA):
    """ Calculate information from mean/covar """
    infoA = np.linalg.inv(covarA)
    hA = np.dot(infoA, mean_A)
    return infoA, hA


@nb.jit(["float64[:, :](float64[:, :], float64[:, :], float64, float64)"], nopython=True, fastmath=True)
def update_info(infoA, zi, addrm, sigma_x):
    """ Add/remove data i to/from information """
    return infoA + addrm * ((1 / sigma_x ** 2) * np.dot(zi.T, zi))


@nb.jit(["float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64, float64)"], nopython=True, fastmath=True)
def update_h(hA, zi, xi, addrm, sigma_x):
    """ Add/remove data i to/from h"""
    return hA + addrm * ((1 / sigma_x ** 2) * np.dot(zi.T, xi))


@nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :]), nopython=True, fastmath=True)
def fromInfo(info_A, hA):
    """ Calculate mean/covar from information """
    covar_A = np.linalg.inv(info_A)
    mean_A = np.dot(covar_A, hA)
    return mean_A, covar_A


@nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64),
        nopython=True, fastmath=True)
def like_xi(zi, mean_A, covarA, sigma_x):
    """ Mean/covar of xi given posterior over A """
    meanXi = np.dot(zi, mean_A)
    covarXi = np.dot(zi, np.dot(covarA, zi.T)) + sigma_x ** 2
    return meanXi, covarXi


@nb.jit(["float64(float64[:, :], float64[:, :], float64[:, :])"], nopython=True, fastmath=True)
def log_p_xi(meanLike, covarLike, xi):
    """
    Calculate log-likelihood of a single xi, given its
    mean/covar after collapsing P(A | X_{-i}, Z)
    """
    D = float(xi.shape[1])
    ll = -(D / 2) * np.log(covarLike)
    ll -= (1 / (2 * covarLike)) * np.power(xi - meanLike, 2).sum()
    return ll.item()


@nb.jit(["int64(float64, float64)"], nopython=True, fastmath=True)
def logBern(lp0, lp1):
    """ Bernoulli sample given log(p0) and log(p1) """
    p1 = 1 / (1 + np.exp(lp0 - lp1))
    return p1 > nr.uniform(0, 1)


@nb.jit(["float64(int64, float64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64)"], nopython=True,
        fastmath=True)
def vLogPost(k, v, mean_A, covarA, xi, zi, sigma_x):
    """ For a given V, calculate the log-posterior """
    oldv = zi[0, k]
    zi[0, k] = v
    (meanLike, covarLike) = like_xi(zi, mean_A, covarA, sigma_x)
    logprior = -0.5 * (v ** 2) - 0.5 * np.log(2 * np.pi)
    loglike = log_p_xi(meanLike, covarLike, xi)
    # Restore previous value and return result
    zi[0, k] = oldv
    return logprior + loglike


@nb.jit(nb.types.UniTuple(nb.float64, 2)(nb.float64, nb.int64, nb.float64, nb.float64[:, :], nb.float64[:, :],
                                         nb.float64[:, :], nb.float64[:, :], nb.float64), nopython=True, fastmath=True)
def makeInterval(u, k, v, mean_A, covarA, xi, zi, sigma_x):
    """ Get horizontal slice sampling interval """
    w = .25
    (left, right) = (v - w, v + w)
    (leftval, rightval) = (vLogPost(k, left, mean_A, covarA, xi, zi, sigma_x),
                           vLogPost(k, right, mean_A, covarA, xi, zi, sigma_x))
    while leftval > u:
        left -= w
        leftval = vLogPost(k, left, mean_A, covarA, xi, zi, sigma_x)
    while rightval > u:
        right += w
        rightval = vLogPost(k, right, mean_A, covarA, xi, zi, sigma_x)
    return left, right


#@nb.jit(nopython=True, fastmath=True)
def _logIBP(ZV, alpha, m):
    """ Calculate IBP prior contribution log P(Z|alpha) """
    (N, K) = ZV.shape
    # Need to find all unique K 'histories'
    Z = (ZV != 0).astype(int)
    Khs = {}
    for k in range(K):
        history = tuple(Z[:, k])
        Khs[history] = Khs.get(history, 0) + 1
    logp = 0
    logp += K * np.log(alpha)
    for Kh in Khs.values():
        logp -= gammaln(Kh + 1)
    logp -= alpha * sum([float(1) / i for i in range(1, N + 1)])
    for k in range(K):
        logp += gammaln(N - m[k] + 1) + gammaln(m[k])
        logp -= gammaln(N + 1)
    if (logp == float('inf')):
        raise Exception
    return logp


@nb.jit(["float64(float64[:, :])"], nopython=True, fastmath=True)
def _logPV(ZV):
    """ Log-likelihood of real-valued latent features V """
    lpv = -0.5 * np.power(ZV, 2).sum()
    return lpv - len(ZV.nonzero()[0]) * 0.5 * np.log(2 * np.pi)


@nb.jit(["float64(int64, float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64)"], nopython=True,
        fastmath=True)
def sample_V(k, mean_A, covarA, xi, zi, sigma_x):
    """ Slice sampling for feature weight V """
    # oldv = zi[0, k]
    # Log-posterior of current value
    curlp = vLogPost(k, zi[0, k], mean_A, covarA, xi, zi, sigma_x)
    # Vertically sample beneath this value
    curval = logUnif(curlp)
    # Initial sample from horizontal slice
    (left, right) = makeInterval(curval, k, zi[0, k], mean_A, covarA, xi, zi, sigma_x)
    newv = nr.uniform(left, right)
    newval = vLogPost(k, newv, mean_A, covarA, xi, zi, sigma_x)
    # Repeat until valid sample obtained
    while (newval <= curval):
        if (newv < zi[0, k]):
            left = newv
        else:
            right = newv
        newv = nr.uniform(left, right)
        newval = vLogPost(k, newv, mean_A, covarA, xi, zi, sigma_x)
    return newv


@nb.jit(nopython=True, fastmath=True)
def sample_Z(N, X, ZV, K, D, m, alpha, sigma_x, sigma_a, use_V):
    """ Take single sample of latent features Z """
    # for each data point
    order = nr.permutation(N)
    for (ctr, i) in enumerate(order):
        # Initially, and later occasionally,
        # re-calculate information directly
        if ctr % 5 == 0:  # DB: <- what is 5?
            mean_A, covar_A = _postA(X, ZV, K, sigma_x, sigma_a)
            info_A, hA = to_info(mean_A, covar_A)
        # Get (z,x) for this data point
        zi = np.reshape(ZV[i, :], (1, K))
        xi = np.reshape(X[i, :], (1, D))
        # xi = X[i:i+1, :]
        # Remove this point from information
        info_A = update_info(info_A, zi, -1, sigma_x)
        hA = update_h(hA, zi, xi, -1, sigma_x)
        # Convert back to mean/covar
        mean_A, covar_A = fromInfo(info_A, hA)
        # Remove this data point from feature cts
        newcts = m - (ZV[i, :] != 0).astype(np.int64)
        # Log collapsed Beta-Bernoulli terms
        lpz1 = np.log(newcts)
        lpz0 = np.log(N - newcts)
        # Find all singleton features
        singletons = [ki for ki in range(K) if
                      ZV[i, ki] != 0 and m[ki] == 1]
        nonsingletons = [ki for ki in range(K) if
                         ki not in singletons]
        # Sample for each non-singleton feature
        #
        for k in nonsingletons:
            oldz = zi[0, k]
            # z=0 case
            lp0 = lpz0[k]
            zi[0, k] = 0.0  # remove element from Z?
            meanLike, covarLike = like_xi(zi, mean_A, covar_A, sigma_x)
            lp0 += log_p_xi(meanLike, covarLike, xi)
            # z=1 case
            lp1 = lpz1[k]
            if use_V:
                if oldz != 0:
                    # Use current V value
                    zi[0, k] = oldz
                    meanLike, covarLike = like_xi(zi, mean_A, covar_A, sigma_x)
                    lp1 += log_p_xi(meanLike, covarLike, xi)
                else:
                    # Sample V values from the prior to
                    # numerically collapse/integrate
                    nvs = 5  # DB: <- what is 5?
                    lps = np.zeros((nvs,))
                    for vs in range(nvs):
                        zi[0, k] = nr.normal(0, 1)
                        (meanLike, covarLike) = like_xi(zi, mean_A, covar_A, sigma_x)
                        lps[vs] = log_p_xi(meanLike, covarLike, xi)
                    lp1 += lps.mean()
            else:
                zi[0, k] = 1.0
                meanLike, covarLike = like_xi(zi, mean_A, covar_A, sigma_x)
                lp1 += log_p_xi(meanLike, covarLike, xi)
            # Sample Z, update feature counts
            if not logBern(lp0, lp1):  # DB: <- rejection sampler?
                zi[0, k] = 0.0
                if oldz != 0:
                    m[k] -= 1
            else:
                if oldz == 0:
                    m[k] += 1
                if use_V:
                    # Slice sample V from posterior if necessary
                    zi[0, k] = 1.0 * sample_V(k, mean_A, covar_A, xi, zi, sigma_x)
        #
        # Sample singleton/new features using the
        # Metropolis-Hastings step described in Meeds et al
        #
        k_old = len(singletons)
        # Sample from the Metropolis proposal
        k_new = nr.poisson(alpha / N)
        if use_V:
            vnew = nr.normal(0, 1, size=k_new)
        # Net difference in number of singleton features
        netdiff = k_new - k_old
        # Contribution of singleton features to variance in x
        if use_V:
            _arr_zi_single = np.asarray([zi[0, s] for s in singletons])
            prevcontrib = np.power(_arr_zi_single, 2).sum()
            newcontrib = np.power(vnew, 2).sum()
            weightdiff = newcontrib - prevcontrib
        else:
            weightdiff = k_new - k_old
        # Calculate the loglikelihoods
        meanLike, covarLike = like_xi(zi, mean_A, covar_A, sigma_x)
        lpold = log_p_xi(meanLike, covarLike, xi)
        lpnew = log_p_xi(meanLike,
                       covarLike + weightdiff * sigma_a ** 2,
                       xi)
        lpaccept = min(0.0, lpnew - lpold)
        lpreject = np.log(max(1.0 - np.exp(lpaccept), 1e-100))
        if logBern(lpreject, lpaccept):
            # Accept the Metropolis-Hastings proposal
            if netdiff > 0:
                # We're adding features, update ZV
                ZV = np.append(ZV, np.zeros((N, netdiff)), 1)
                if use_V:
                    prev_num_singletons = len(singletons)
                    for k, s in enumerate(singletons):
                        ZV[i, s] = 1.0 * vnew[k]
                    # ZV[i, singletons] = vnew[:prev_num_singletons]
                    ZV[i, K:] = vnew[prev_num_singletons:]
                else:
                    ZV[i, K:] = 1.0
                # Update feature counts m
                m = np.append(m, np.ones(netdiff, dtype=np.int32), 0)
                # Append information matrix with 1/sigmaa^2 diag
                info_A = np.vstack((info_A, np.zeros((netdiff, K))))
                info_A = np.hstack((info_A,
                                    np.zeros((netdiff + K, netdiff))))
                infoappend = (1 / sigma_a ** 2) * np.eye(netdiff)
                info_A[K:(K + netdiff),
                K:(K + netdiff)] = infoappend
                # only need to resize (expand) hA
                hA = np.vstack((hA, np.zeros((netdiff, D))))
                # Note that the other effects of new latent features 
                # on (info_A,hA) (ie, the zi terms) will be counted when
                # this zi is added back in                    
                K += netdiff
            elif netdiff < 0:
                # We're removing features, update ZV
                if use_V:
                    for k, s in enumerate(singletons[(-1 * netdiff):]):
                        ZV[i, int(s)] = 1.0 * vnew[k]
                    # ZV[i, singletons[(-1 * netdiff):]] = vnew
                dead = [ki for ki in singletons[:(-1 * netdiff)]]
                K -= len(dead)
                # delete rows/columns from Z
                ZV_temp = np.zeros((ZV.shape[0], ZV.shape[1] - len(dead)), dtype=np.float64)
                k = 0
                for i in range(ZV.shape[1]):
                    if i not in dead:
                        ZV_temp[:, k] = ZV[:, i]
                        k += 1
                ZV = ZV_temp.copy()
                m = np.delete(m, dead)
                # Easy to do this b/c these features did not
                # occur in any other data points anyways...
                info_A_temp = np.zeros((info_A.shape[0] - len(dead), info_A.shape[1]))
                k = 0
                for i in range(info_A.shape[0]):
                    if i not in dead:
                        info_A_temp[k] = info_A[i]
                        k += 1
                info_A = info_A_temp.copy()
                info_A_temp = np.zeros((info_A.shape[0], info_A.shape[1] - len(dead)))
                k = 0
                for i in range(info_A.shape[1]):
                    if i not in dead:
                        info_A_temp[:, k] = info_A[:, i]
                        k += 1
                info_A = info_A_temp.copy()

                hA_temp = np.zeros((hA.shape[0] - len(dead), hA.shape[1]))
                k = 0
                for i in range(hA.shape[0]):
                    if i not in dead:
                        hA_temp[k] = hA[i]
                        k += 1
                hA = hA_temp.copy()
            else:
                # net difference is actually zero, just replace
                # the latent weights of existing singletons
                # (if applicable)
                if use_V:
                    for k, s in enumerate(singletons):
                        ZV[i, s] = 1.0 * vnew[k]

        # Add this point back into information
        # DB: <- do we need this?
        zi = np.reshape(ZV[i, :], (1, K))
        info_A = update_info(info_A, zi, 1, sigma_x)
        hA = update_h(hA, zi, xi, 1, sigma_x)

    return ZV, K, m
