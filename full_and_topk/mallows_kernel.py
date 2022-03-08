import numpy as np

from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from sympy.combinatorics import Permutation
from sklearn.metrics import pairwise_distances


def kernel_mallows(sigma,sigma_prime,lamb):
    eta = Permutation(sigma_prime)* ~Permutation(sigma)
    discordant_pairs = eta.inversions()
    if lamb is None:
        Mall = np.exp((-1)*discordant_pairs)
    else:
        Mall = np.exp(-lamb*discordant_pairs)
    return Mall


def compute_kernel_matrix(my_kernel, my_kernel_args, X, Y):
    return pairwise_distances(X, Y, lambda u, v: my_kernel(u, v, my_kernel_args))

def top_k_sample(topk, n_deg):
    if topk is None:
        result = np.random.permutation(n_deg)
    else:
        items = list(range(n_deg))
        remaining_items = []
        for item in items:
            if item not in set(topk):
                remaining_items.append(item)
        if len(remaining_items) > 1:
            remaining_items = np.random.permutation(remaining_items)
        # Construct output ranking
        result = list(topk)+list(remaining_items)

    return result

def get_complete_rankings(partial_rankings, n_mc_samples, n_deg):
    complete_rankings = []
    n_rankings = len(partial_rankings)
    for ell in range(n_rankings):
        particles = np.zeros([n_mc_samples, n_deg])
        for i in range(n_mc_samples):
            particles[i, :] = top_k_sample(partial_rankings[ell], n_deg)
        complete_rankings.append(particles)
    return complete_rankings

def approx_gram_mat_topk(my_kernel, mykernargs, partial_rankings, n_mc_samples, n_deg, rank_only = False):

    n_rankings = len(partial_rankings)
    complete_rankings = get_complete_rankings(partial_rankings, n_mc_samples, n_deg)
    if not rank_only:
      K_mat_mc = np.zeros([n_rankings, n_rankings])
      for ell in range(n_rankings):
          for kk in range(n_rankings):
              K_aux = compute_kernel_matrix(my_kernel, mykernargs,
                                            complete_rankings[ell], complete_rankings[kk])
              weights = np.ones(n_mc_samples) / float(n_mc_samples)
              K_mat_mc[ell, kk] = np.dot(weights.T, np.dot(K_aux, weights))
    # Flatten, so that we have a 2D array to pass to GPy
    complete_rankings_flat = np.array([fr.flatten() for fr in complete_rankings])

    return (complete_rankings_flat, K_mat_mc) if not rank_only else complete_rankings_flat



class Mallows(Kern):

    def __init__(self, input_dim, variance=1., scale=1., active_dims=None, name='mallows'):
        super(Mallows, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.scale = Param('scale', scale, Logexp())
        self.link_parameters(self.variance, self.scale)

    def kernel_mallows(self, sigma,sigma_prime):
        eta = Permutation(sigma_prime)* ~Permutation(sigma)
        discordant_pairs = eta.inversions()
        Mall = np.exp(-self.scale*discordant_pairs)
        return Mall

    def K(self, X, X2=None):
        return self.variance * pairwise_distances(X, X2, lambda u, v: self.kernel_mallows(u,v))

    def Kdiag(self, X):
        return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        #raise NotImplementedError
        self.scale.gradient = 0
        self.variance.gradient = 0

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError


class Mallows_PRMC(Kern):
    """
    Monte Carlo approx to Mallows kernel for partial rankings
    """

    def __init__(self, input_dim, variance=1., scale=1.,ndeg=1, active_dims=None, name='PRMC-mallows'):
        super(Mallows_PRMC, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.scale = Param('scale', scale, Logexp())
        # self.link_parameters(self.variance, self.scale)

    def kernel_pr(self, sigmas, sigmaprimes):
        # First, unpack
        ndeg = int(max(sigmas)+1)
        n_fullrankings = len(sigmas)//ndeg
        assert type(n_fullrankings) is int, 'dimension of array is {} and ndeg is {}'.format(np.shape(sigmas)[1], ndeg)
        sigmas_unf = np.reshape(sigmas,((n_fullrankings, ndeg)))
        n_fullrankings_prime = len(sigmaprimes)//ndeg
        assert type(n_fullrankings_prime) is int, 'dimension of array is {} and ndeg is {}'.format(np.shape(sigmas)[1], ndeg)
        sigmaprimes_unf = np.reshape(sigmaprimes,((n_fullrankings_prime, ndeg)))
        n_sigmas = len(sigmas_unf)
        n_sigmaprimes = len(sigmaprimes_unf)
        ker_val = 0
        for ii in range(n_sigmas):
            for jj in range(n_sigmaprimes):
                if set(sigmas_unf[ii,: ]) != set(range(ndeg)):
                    raise ValueError('This is not a valid permutation of ndeg')
                ker_val += self.kernel_mallows(sigmas_unf[ii, :], sigmaprimes_unf[jj,:])
        ker_val /= (n_sigmas * n_sigmaprimes)
        return ker_val

    def kernel_mallows(self, sigma, sigma_prime):
        eta = Permutation(sigma_prime)* ~Permutation(sigma)
        discordant_pairs = eta.inversions()
        Mall = np.exp(-self.scale*discordant_pairs)
        return Mall

    def K(self, X, X2=None):
        return self.variance * pairwise_distances(X, X2, lambda u, v: self.kernel_pr(u,v))

    def Kdiag(self, X):
        return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        #raise NotImplementedError
        self.scale.gradient = 0
        self.variance.gradient = 0

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError


import GPy

from GPy.util.linalg import tdot
from GPy.util.diag import view

class MallowsGPy(GPy.kern.RBF):

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False,
                 active_dims=None, name='Mallows', useGPU=False, inv_l=False):
        super(MallowsGPy, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)

    ### CUSTOM FOR MALLOWS ###
    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2 / 4) * 2
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2 / 4) * 2
    ###########################


import tensorflow as tf
import gpflow
from scipy.special import comb


def mallow_dist(X, X2):
    r2 = gpflow.utilities.ops.square_distance(X, X2)
    r2 = tf.clip_by_value(r2, 0, tf.math.reduce_max(r2))
    return r2 / 4

    # return tf.sqrt(r2 / 4)  # *2 is necessary because RBF is defined as 0.5 * r^2

class N_discord_distance(gpflow.kernels.Stationary):
    def __init__(self, base: gpflow.kernels.Stationary):
        self.base = base

    @property
    def active_dims(self):
        return self.base.active_dims

    @property
    def variance(self):
        return self.base.variance  # for K_diag to work

    def K(self, X, X2=None, presliced=False):
        if not presliced: X, X2 = self.slice(X, X2)
        if X2 is None:    X2 = X
        r = mallow_dist(X, X2) / self.base.lengthscales
        return self.base.K_r2( 2 * r )


MallowsFlow = lambda ls : N_discord_distance(gpflow.kernels.RBF(lengthscales=ls))

def calc_phi(s, d):

  ss = np.zeros(len(s))
  for i,e in enumerate(s):
    ss[e] = len(s)-i
  s = ss

  sigma_phi = np.zeros(d)
  k = 0
  for j in range(1, len(s)):
    for i in range(j):
      sigma_phi[k] = int(s[i]>s[j]) - int(s[i]<s[j])
      k+=1
  return sigma_phi


def get_phi_mallows(X, N_objs, train_ind, test_ind):
  d = int(comb(N_objs,2))
  X_phi = np.zeros((len(X), d), dtype = np.float64)
  for i,x in enumerate(X):
    X_phi[i] = calc_phi(x, d)
  return X_phi[train_ind.ravel()], X_phi[test_ind.ravel()]



class Mallows_PRMCflow(gpflow.kernels.Stationary):
    def __init__(self, ls = 1, n_mc_samples = 4):
        self.base = gpflow.kernels.RBF(lengthscales=ls)
        self.n_mc_samples = n_mc_samples

    @property
    def active_dims(self):
        return self.base.active_dims
    @property
    def variance(self):
        return self.base.variance  # for K_diag to work

    def get_b(self, N):
        b = np.zeros((N,N//self.n_mc_samples))
        last_i = 0
        j = 0
        for i in range(self.n_mc_samples, b.shape[0], self.n_mc_samples):
          b[last_i:i,j] = 1
          j += 1
          last_i = i
        b[last_i:,j] = 1
        return tf.constant(b)

    def K(self, X, X2=None, presliced=False):

        # if not presliced: X, X2 = self.slice(X, X2)

        X = tf.reshape(X, (len(X) * self.n_mc_samples, -1))

        if not X2 is None:
          X2 = tf.reshape(X2, (len(X2) * self.n_mc_samples, -1))

        r = mallow_dist(X, X2) / self.base.lengthscales

        b = tf.transpose(self.get_b(len(X)))
        if not X2 is None:  b2 = self.get_b(len(X2))
        else:               b2 = self.get_b(len(X))

        return (b @ self.base.K_r2( 2 * r ) @ b2) / self.n_mc_samples**2


def get_phi_topk(X, n_mc_samples, N_objs, train_ind, test_ind):
    X_completerankings = approx_gram_mat_topk(kernel_mallows,1,  X, n_mc_samples, N_objs, True).astype(int)
    X_completerankings = X_completerankings.reshape(len(X) * n_mc_samples, -1)
    X_, _ = get_phi_mallows(X_completerankings, N_objs, np.arange(len(X_completerankings)),
                                       np.arange(3))
    X_ = X_.reshape(len(X_) // n_mc_samples, -1)
    return X_[train_ind.ravel()], X_[test_ind.ravel()]



























# def n_disc(sigma, sigma_prime):
#     eta = Permutation(sigma_prime)* ~Permutation(sigma)
#     discordant_pairs = eta.inversions()
#     return discordant_pairs


# from kendall_kernel import KendallFull








# sigmas                 = X_completerankings[0]
# sigmaprimes = sigma_p = X_completerankings[1]




# self = K






# d = int(comb(len(sigma),2))
# def calc_phi(s):

#   ss = np.zeros(len(s))
#   for i,e in enumerate(s):
#     ss[e] = len(s)-i
#   s = ss

#   sigma_phi = np.zeros(d)
#   k = 0
#   for j in range(1, len(sigma)):
#     for i in range(j):
#       sigma_phi[k] = int(s[i]>s[j]) - int(s[i]<s[j])
#       k+=1
#   return sigma_phi

# sigma_phi = calc_phi(sigma)
# sigma_prime_phi = calc_phi(sigma_prime)



# sigma_phi.dot(sigma_prime_phi)



# n_disc(sigma, sigma_prime)
# round(np.linalg.norm(sigma_phi - sigma_prime_phi)**2,1) / 4



# kkk = KendallFull(5,5)

# round(2-2*kkk.kernel_kendall(np.array(sigma), np.array(sigma_prime)),1) / 4 * d



# X_phi, _ = get_phi_mallows(X, 10, np.arange(100),np.arange(100,150))


# X_n_disc = np.zeros((len(X_phi),len(X_phi)))
# for i in range(len(X_n_disc)):
#   for j in range(i, len(X_n_disc)):
#     X_n_disc[i,j] = X_n_disc[j,i] = n_disc(X[i],X[j])





# KK = MallowsFlow(1)
# KK.K(X_phi).numpy().round(2)



# np.exp(-mallow_dist(X_phi,  None)).round(2)

