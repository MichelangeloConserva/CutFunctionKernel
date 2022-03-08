import gpflow
import tensorflow as tf
import numpy as np

from scipy.special import comb
from GPy.kern import Kern
from sklearn.metrics import pairwise_distances

try:
  from rpy2.robjects.packages import importr
  import rpy2.robjects as ro
  kernrank = importr('kernrank')
except:
  raise ImportError("Make sure that rpy2 is installed and that kernrnak packgage is installed in R")


class KendallTOPk(Kern):

    def converter(self, sigma):
      s = ""
      for o in range(self.N_objs):
        if o in sigma:
          s += str(-sigma.tolist().index(o))
        else:
          s += "NA"
        s+=","
      s = s[:-1]
      return ro.r("c("+s+")")

    def __init__(self, input_dim, N_objs, variance=1., active_dims=None, name='kendall'):
        super(KendallTOPk, self).__init__(input_dim, active_dims, name)
        self.variance = variance
        self.N_objs = N_objs

    def kernel_kendall(self, sigma,sigma_prime):
        sigma = self.converter(sigma)
        sigma_ = self.converter(sigma_prime)
        return  kernrank.kendall_top(sigma,sigma_)[0]

    def K(self, X, X2=None):
        return self.variance * pairwise_distances(X, X2, lambda u, v: self.kernel_kendall(u,v))

    def Kdiag(self, X): return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None): self.variance.gradient = 0
    def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError


class KendallPartialflow(gpflow.kernels.base.Kernel):
  def __init__(self, K, N_objs, variance = 1.0):
      super().__init__(None)

      self.K_ = tf.constant(K)
      self.variance = gpflow.base.Parameter(variance, transform=gpflow.utilities.positive())

  def K(self, X, X2):
    X = tf.cast(tf.squeeze(X), tf.int32)

    if X2 is None: X2 = X
    else:  X2 = tf.cast(tf.squeeze(X2), tf.int32)

    return self.variance * tf.gather(tf.gather(self.K_, X),X2, axis =1)

  def K_diag(self, X):
      return tf.linalg.diag_part(self.K(X, None))


class KendallInterLeaving(Kern):

    def converter(self, sigma):
      s = [""] * self.N_objs
      for i,sb in enumerate(sigma):
        for o in sb:
          s[o] = str(-i)
      s = ",".join(s)
      return ro.r("c("+s+")")

    def __init__(self, input_dim, N_objs, variance=1., active_dims=None, name='kendall'):
        super(KendallInterLeaving, self).__init__(input_dim, active_dims, name)
        self.variance = variance
        self.N_objs = N_objs

    def kernel_kendall(self, sigma,sigma_prime):
        sigma = self.converter(sigma)
        sigma_ = self.converter(sigma_prime)
        return  kernrank.kendall_partial(sigma,sigma_)[0]

    def K(self, X, X2=None):
        return self.variance * pairwise_distances(X, X2, lambda u, v: self.kernel_kendall(u,v))

    def Kdiag(self, X): return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None): self.variance.gradient = 0
    def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError


class KendallInterLeavingNonEx(Kern):

    def converter(self, sigma):
      s = ["NA"] * self.N_objs
      for i,ss in enumerate(sigma):
        for sss in ss: s[sss] = str(-i)
      return ro.r("c("+",".join(s)+")")

    def __init__(self, input_dim, N_objs, variance=1., active_dims=None, name='kendall'):
        super(KendallInterLeavingNonEx, self).__init__(input_dim, active_dims, name)
        self.variance = variance
        self.N_objs = N_objs

    def kernel_kendall(self, sigma,sigma_prime):
        sigma = self.converter(sigma)
        sigma_ = self.converter(sigma_prime)
        return  kernrank.kendall_partial(sigma,sigma_)[0]

    def K(self, X, X2=None):
        return self.variance * pairwise_distances(X, X2, lambda u, v: self.kernel_kendall(u,v))

    def Kdiag(self, X): return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None): self.variance.gradient = 0
    def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError


class KendallFull(Kern):

    def converter(self, sigma):
      s = ""
      for o in range(self.N_objs):
        if o in sigma:
          s += str(-sigma.tolist().index(o))
        else:
          raise ValueError("This should not happen for FULL kendall")
        s+=","
      s = s[:-1]
      return ro.r("c("+s+")")

    def __init__(self, input_dim, N_objs, variance=1., active_dims=None, name='kendall'):
        super(KendallFull, self).__init__(input_dim, active_dims, name)
        self.variance = variance
        self.N_objs = N_objs

    def kernel_kendall(self, sigma,sigma_prime):
        sigma = self.converter(sigma)
        sigma_ = self.converter(sigma_prime)
        return  kernrank.kendall_total(sigma,sigma_)[0]

    def K(self, X, X2=None):
      return pairwise_distances(X, X2, self.kernel_kendall)

    def Kdiag(self, X): return self.K(X).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None): self.variance.gradient = 0
    def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError


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
  return sigma_phi / np.sqrt(d)


def get_phi_kendall(X, N_objs, train_ind, test_ind):
  d = int(comb(N_objs,2))
  X_phi = np.zeros((len(X), d), dtype = np.float64)
  for i,x in enumerate(X):
    X_phi[i] = calc_phi(x, d)
  return X_phi[train_ind.ravel()], X_phi[test_ind.ravel()]

