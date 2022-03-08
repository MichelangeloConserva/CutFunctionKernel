import networkx as nx
import numpy as np

from GPy.kern import Kern
from functools import reduce

class Ours(Kern):

    def __init__(self, input_dim, G, scale=1., variance = 1,
                 active_dims=None, name='ours', K_matrix = None, kern = "linear",
                 pava = True):
        Kern.__init__(self, input_dim, active_dims, name)

        self.scale = scale
        self.N_objs = input_dim
        self.G = G
        self.K_matrix = K_matrix
        self.kern = kern
        self.variance = variance
        self.pava = pava


    def kernel_f(self, sigma, sigma_prime):
        if self.kern == "exp":
            return np.exp(-self.scale*np.linalg.norm(self.phi_(sigma) - self.phi_(sigma_prime)))
        elif self.kern == "linear":
            return np.dot(self.phi_(sigma), self.phi_(sigma_prime))
        elif self.kern == "linear*exp":
            phi_sigma = self.phi_(sigma)
            phi_sigma_prime = self.phi_(sigma_prime)

            l = np.dot(phi_sigma, phi_sigma_prime)
            e = np.exp(-self.scale*np.linalg.norm(phi_sigma - phi_sigma_prime))
            return l*e

    def _index(self, X, X2):
        if X2 is None: i1 = i2 = X.astype('int').flat
        else:          i1, i2 = X.astype('int').flat, X2.astype('int').flat
        return self.K_matrix[i1,:][:,i2]

    def K(self, X, X2=None): return self.variance * self._index(X, X2)
    def Kdiag(self, X): return self.variance * self._index(X,None).diagonal()
    def update_gradients_full(self, dL_dK, X, X2=None): pass
    def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError

    def calc_v(self, groups):
        v = np.zeros(len(groups))
        B_i, B_i_ = set(), set()
        k = 0
        while len(B_i) < self.N_objs:
            B_i = B_i.union(groups[len(groups) - 1 - k])
            # B_i = B_i.union(groups[k])
            v[k] = - (self.F(B_i) - self.F(B_i_)) / (len(B_i)-len(B_i_))
            B_i_ = B_i.copy()
            k += 1
        return v

    def F(self, A_): return nx.cut_size(self.G, A_, None, 'weight')

    def phi_(self, A):
        assert type(A[0]) == set
        A_is = A.copy()

        if not self.pava:
          v = self.calc_v(A_is)
        else:
          v = []
          k = len(A_is)
          while len(v) < len(A_is):

              B_i  = reduce(lambda a,b: a.union(b), A_is[k-1:])
              B_i_ = reduce(lambda a,b: a.union(b), A_is[k:]) if k < len(A_is) else set([])
              v_ =  - (self.F(B_i) - self.F(B_i_)) / (len(B_i)-len(B_i_))


              if len(v) != 0 and v_ < v[0]:
                  A_is[k-1:k+1] = [A_is[k-1].union(A_is[k])]
                  v.pop(0)
                  continue


              v.insert(0,v_)
              k -= 1

        w = np.zeros(self.N_objs)
        # Reordering
        for i in range(len(A_is)):  w[list(A_is[i])] = v[i]

        # Not Reordering
        # for a,i in zip(A_is,range(len(v))):
        #   w[list(a)] = v[i]

        return - w




def get_phi(X_ours, G, N_objs, train_ind, test_ind):
  X_phi = np.zeros((len(X_ours), N_objs))
  kern = Ours(N_objs, G, scale = 1, variance = 1, kern="linear*exp", pava = True)
  for i,x in enumerate(X_ours):
    X_phi[i] = kern.phi_(x) #/ np.linalg.norm(kern.phi_(x))
  return X_phi[train_ind.ravel()], X_phi[test_ind.ravel()]