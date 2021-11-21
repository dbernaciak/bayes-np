import scipy.stats as st
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import asyncio
import nest_asyncio

nest_asyncio.apply()


@nb.jit(nopython=True)
def logpdf(x, mu, cov):
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


class GaussianMixture:
    def __init__(self, sd, rho, mu=None):
        """
        :param sd: covariance matrix of data points around the, i.e. this is the standard deviation in either direction
        :param rho: vector of component frequencies
        :param mu: cluster-specific mean is [sd^2, 0; 0, sd^2]
        """
        self.sd = sd
        self.rho = np.asarray(rho) / sum(rho)
        self.mu = np.array([[3, 3], [-3, 3], [3, -3], [-3, -3]]) if mu is None else mu
        self.z = None

    def rvs(self, size):
        """
        :param size:
        :return:
        """

        # assign each data point to a component
        z = np.random.choice(np.arange(0, len(self.rho), 1), size=size, replace=True, p=self.rho)
        self.z = z
        # draw each data point according to the cluster-specific
        # likelihood of its component
        # an Ndata x 2 matrix of data points
        x = np.array([st.norm(self.mu[z, 0], self.sd).rvs(size=size),
                      st.norm(self.mu[z, 1], self.sd).rvs(size=size)])
        return x.T


@nb.jit(nopython=True)
def run_mcmc(data: np.ndarray, alpha: float = None, max_iter: int = 1000, sd: float = 1, sig0: float = 3.0):
    """
    Fast implementation of the algorithm 3 of Neal (2000) with extension to hyperparameter inference for alpha

    :param data:
    :param alpha:
    :param max_iter:
    :param sd:
    :param sig0: prior covariance
    :return:
    """

    final_loc_probs = {}
    data_dim = data.shape[1]  # dimension of the data points
    sig = np.eye(data_dim) * sd ** 2  # cluster-specific covariance matrix
    sig0 = np.eye(data_dim) * sig0 ** 2  # prior covariance matrix TODO: fix me
    prec = np.linalg.inv(sig)
    prec0 = np.linalg.inv(sig0)
    mu0 = np.array([0.0, 0.0])  # prior mean on cluster parameters
    ndata = int(data.shape[0])  # number of data points
    z = np.zeros(data.shape[0], dtype=np.int64)  # initial cluster assignments
    counts = np.array([data.shape[0]])
    n_clusters = len(counts)
    pi_choice = np.random.uniform(0.0, 1.0, size=max_iter)
    infer_alpha = True if alpha is None else False
    alpha = 0.01 if alpha is None else alpha
    a_gamma = 2
    b_gamma = 4
    alphas = np.empty(max_iter)
    for it in range(max_iter):
        for n in range(ndata):
            c = int(z[n])
            counts[c] -= 1
            if counts[c] == 0:
                counts[c] = counts[n_clusters - 1]
                loc_c = np.argwhere(z == n_clusters - 1).ravel()
                for loc in loc_c:
                    z[loc] = c
                counts = np.delete(counts, n_clusters - 1)
                n_clusters -= 1

            z[n] = -1  # ensures z[n] doesn't get counted as a cluster
            log_weights = np.empty(n_clusters + 1)
            # find the unnormalized log probabilities
            # for each existing cluster
            for c in range(n_clusters):
                c_precision = prec0 + counts[c] * prec
                c_sig = np.linalg.inv(c_precision)  # np.eye(data_dim) * 1 / np.diag(c_precision)  #
                loc_z = np.where(z == int(c))[0]
                if len(loc_z) > 1:
                    sum_data = np.sum(data[z == c, :], axis=0)
                else:
                    sum_data = data[z == c, :].ravel()

                c_mean = c_sig @ (prec @ sum_data.T + prec0 @ mu0.T)
                log_weights[c] = np.log(counts[c]) + logpdf(data[n, :], c_mean, c_sig + sig)

            # find the unnormalized log probability
            # for the "new" cluster
            log_weights[n_clusters] = np.log(alpha) + logpdf(data[n, :], mu0, sig0 + sig)
            # transform unnormalized log probabilities
            # into probabilities
            max_weight = np.max(log_weights)
            log_weights = log_weights - max_weight
            loc_probs = np.exp(log_weights)
            loc_probs = loc_probs / sum(loc_probs)

            # sample which cluster this point should belong to
            newz = np.argwhere(np.random.multinomial(1, loc_probs, size=1).ravel() == 1)[0][0]
            # newz = np.random.choice(np.arange(0, n_clusters + 1, 1), 1, replace=True, p=loc_probs)
            # if necessary, instantiate a new cluster
            if newz == n_clusters:
                counts = np.append(counts, 0)
                n_clusters += 1

            z[n] = newz
            # update the cluster counts
            counts[newz] += 1

            if it == max_iter - 1:
                final_loc_probs[n] = loc_probs

        alphas[it] = alpha
        if infer_alpha:
            z_temp = np.random.beta(alpha + 1, ndata)
            pi1 = a_gamma + n_clusters - 1
            pi2 = ndata * (b_gamma - np.log(z_temp))
            pi_prob = pi1 / (pi1 + pi2)

            if pi_prob >= pi_choice[it]:
                alpha = np.random.gamma(a_gamma + n_clusters, 1 / (b_gamma - np.log(z_temp)))
            else:
                alpha = np.random.gamma(a_gamma + n_clusters - 1, 1 / (b_gamma - np.log(z_temp)))

    return z, final_loc_probs, alphas


class CRPGibbs:
    def __init__(self, data, sd, initz=None):
        """

        :param data:
        :param sd:
        :param initz:
        """
        self.data = data
        self.sd = sd
        self.initz = initz if initz is not None else np.zeros(data.shape[0])
        self.z = None
        self.loc_probs = {}

    def run(self, alpha, max_iter=1000):
        """

        :param alpha:
        :param max_iter:
        :return:
        """
        loop = asyncio.get_event_loop()
        coroutine = self.run_live(alpha, max_iter=max_iter, is_live=False)
        loop.run_until_complete(coroutine)

    async def run_live(self, alpha, max_iter=1000, is_live=True, fig=None):
        """

        :param alpha:
        :param max_iter:
        :return:
        """
        data_dim = self.data.shape[1]  # dimension of the data points
        sig = np.eye(data_dim) * self.sd ** 2  # cluster-specific covariance matrix
        sig0 = np.eye(data_dim) * 3 ** 2  # prior covariance matrix TODO: fix me
        prec = np.linalg.inv(sig)
        prec0 = np.linalg.inv(sig0)
        mu0 = np.array([0.0, 0.0])  # prior mean on cluster parameters
        ndata = self.data.shape[0]  # number of data points
        z = self.initz if self.z is None else self.z  # initial cluster assignments
        colors = ["blue", "orange", "green", "yellow", "red", "purple", "black"]
        if is_live:
            plt.clf()
            for i, dat in enumerate(self.data):
                plt.scatter(dat[0], dat[1], c=colors[int(z[i])])
            fig.canvas.draw()
            await asyncio.sleep(1)
        _, counts = np.unique(z, return_counts=True)
        n_clusters = len(counts)
        for it in tqdm(range(max_iter)):
            for n in range(ndata):
                c = int(z[n])
                counts[c] -= 1
                if counts[c] == 0:
                    counts[c] = counts[n_clusters - 1]
                    loc_c = np.argwhere(z == n_clusters - 1)
                    z[loc_c] = c
                    counts = np.delete(counts, n_clusters - 1)
                    n_clusters -= 1

                z[n] = -1  # ensures z[n] doesn't get counted as a cluster
                log_weights = np.empty(n_clusters + 1)
                # find the unnormalized log probabilities
                # for each existing cluster
                for c in range(n_clusters):
                    c_precision = prec0 + counts[c] * prec
                    c_sig = np.eye(data_dim) * 1 / np.diag(c_precision)  # np.linalg.inv(c_precision)
                    loc_z = np.asarray(np.where(z == c)).ravel()
                    if len(loc_z) > 1:
                        sum_data = np.sum(self.data[z == c, :], axis=0)
                    else:
                        sum_data = self.data[z == c, :].ravel()

                    c_mean = c_sig @ (prec @ sum_data.T + prec0 @ mu0.T)
                    log_weights[c] = np.log(counts[c]) + st.multivariate_normal(c_mean, c_sig + sig).logpdf(
                        self.data[n, :])

                # find the unnormalized log probability
                # for the "new" cluster
                log_weights[n_clusters] = np.log(alpha) + st.multivariate_normal(mu0, sig0 + sig).logpdf(
                    self.data[n, :])
                # transform unnormalized log probabilities
                # into probabilities
                max_weight = np.max(log_weights)
                log_weights = log_weights - max_weight
                loc_probs = np.exp(log_weights)
                loc_probs = loc_probs / sum(loc_probs)

                # sample which cluster this point should belong to
                newz = np.random.choice(np.arange(0, n_clusters + 1, 1), 1, replace=True, p=loc_probs)
                # if necessary, instantiate a new cluster
                if newz == n_clusters:
                    counts = np.append(counts, 0)
                    n_clusters += 1

                z[n] = newz
                # update the cluster counts
                counts[newz] += 1
                if is_live:
                    plt.scatter(self.data[n, 0], self.data[n, 1], c=colors[int(newz)])
                    fig.canvas.draw()
                    await asyncio.sleep(0.001)
                if it == max_iter - 1:
                    self.loc_probs[n] = loc_probs

        self.z = z
        if is_live:
            print("Done!")