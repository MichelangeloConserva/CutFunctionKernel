import networkx as nx
import numpy as np

# from GPy.kern import Kern
from functools import reduce
from itertools import product
import copy

# class Ours(Kern):

#     def __init__(self, input_dim, G, scale=1., variance = 1,
#                  active_dims=None, name='ours', K_matrix = None, kern = "linear",
#                  pava = True):
#         Kern.__init__(self, input_dim, active_dims, name)

#         self.scale = scale
#         self.N_objs = input_dim
#         self.G = G
#         self.K_matrix = K_matrix
#         self.kern = kern
#         self.variance = variance
#         self.pava = pava


#     def kernel_f(self, sigma, sigma_prime):
#         if self.kern == "exp":
#             return np.exp(-self.scale*np.linalg.norm(self.phi_(sigma) - self.phi_(sigma_prime)))
#         elif self.kern == "linear":
#             return np.dot(self.phi_(sigma), self.phi_(sigma_prime))
#         elif self.kern == "linear*exp":
#             phi_sigma = self.phi_(sigma)
#             phi_sigma_prime = self.phi_(sigma_prime)

#             l = np.dot(phi_sigma, phi_sigma_prime)
#             e = np.exp(-self.scale*np.linalg.norm(phi_sigma - phi_sigma_prime))
#             return l*e

#     def _index(self, X, X2):
#         if X2 is None: i1 = i2 = X.astype('int').flat
#         else:          i1, i2 = X.astype('int').flat, X2.astype('int').flat
#         return self.K_matrix[i1,:][:,i2]

#     def K(self, X, X2=None): return self.variance * self._index(X, X2)
#     def Kdiag(self, X): return self.variance * self._index(X,None).diagonal()
#     def update_gradients_full(self, dL_dK, X, X2=None): pass
#     def update_gradients_diag(self, dL_dKdiag, X):      raise NotImplementedError
#     def gradients_X(self, dL_dK, X, X2=None):           raise NotImplementedError
#     def gradients_X_diag(self, dL_dKdiag, X):           raise NotImplementedError

#     def calc_v(self, groups):
#         v = np.zeros(len(groups))
#         B_i, B_i_ = set(), set()
#         k = 0
#         while len(B_i) < self.N_objs:
#             B_i = B_i.union(groups[len(groups) - 1 - k])
#             # B_i = B_i.union(groups[k])
#             v[k] = - (self.F(B_i) - self.F(B_i_)) / (len(B_i)-len(B_i_))
#             B_i_ = B_i.copy()
#             k += 1
#         return v

#     def F(self, A_): return nx.cut_size(self.G, A_, None, 'weight')

#     def phi_(self, A):
#         assert type(A[0]) == set
#         A_is = A.copy()

#         if not self.pava:
#           v = self.calc_v(A_is)
#         else:
#           v = []
#           k = len(A_is)
#           while len(v) < len(A_is):

#               B_i  = reduce(lambda a,b: a.union(b), A_is[k-1:])
#               B_i_ = reduce(lambda a,b: a.union(b), A_is[k:]) if k < len(A_is) else set([])
#               v_ =  - (self.F(B_i) - self.F(B_i_)) / (len(B_i)-len(B_i_))


#               if len(v) != 0 and v_ < v[0]:
#                   A_is[k-1:k+1] = [A_is[k-1].union(A_is[k])]
#                   v.pop(0)
#                   continue


#               v.insert(0,v_)
#               k -= 1

#         w = np.zeros(self.N_objs)
#         # Reordering
#         for i in range(len(A_is)):  w[list(A_is[i])] = v[i]

#         # Not Reordering
#         # for a,i in zip(A_is,range(len(v))):
#         #   w[list(a)] = v[i]

#         return - w


# def F(A_, G): return nx.cut_size(G, A_, None, 'weight')
def F(A_, G):
    return nx.cut_size(G, A_, None)


def phi_(A, N_objs, G):
    assert type(A[0]) == set
    A_is = A.copy()

    v = []
    k = len(A_is)
    while len(v) < len(A_is):

        B_i = reduce(lambda a, b: a.union(b), A_is[k - 1 :])
        B_i_ = reduce(lambda a, b: a.union(b), A_is[k:]) if k < len(A_is) else set([])
        v_ = -(F(B_i, G) - F(B_i_, G)) / (len(B_i) - len(B_i_))

        if len(v) != 0 and v_ < v[0]:
            A_is[k - 1 : k + 1] = [A_is[k - 1].union(A_is[k])]
            v.pop(0)
            continue

        v.insert(0, v_)
        k -= 1

    w = np.zeros(N_objs)
    for i in range(len(A_is)):
        w[list(A_is[i])] = v[i]

    return -w


def get_phi(X_ours, G, N_objs, train_ind, test_ind):
    X_phi = np.zeros((len(X_ours), N_objs))
    for i, x in enumerate(X_ours):
        f = phi_(x, N_objs, G)
        X_phi[i] = f / np.linalg.norm(f)
    return X_phi[train_ind.ravel()], X_phi[test_ind.ravel()]


def phi_interleaving(A_inter, G, N_objs, heuristic=False, samples=2000):

    absents = list(set(list(range(N_objs))) - set(list(A_inter)))
    inter = [set()]
    for o in A_inter:
        inter.append(set(([o])))
        inter.append(set())
    possible_positions = list(range(0, len(inter), 2))
    X_inter_phi = np.zeros(N_objs)

    if not heuristic or samples >= len(possible_positions) ** len(absents):
        coherent_set = product(possible_positions, repeat=len(absents))
        div = len(possible_positions) ** len(absents)
    else:
        rng = np.random.RandomState(N_objs)
        coherent_set = rng.choice(possible_positions, (samples, len(absents)))
        div = samples

    for i, abs_pos in enumerate(coherent_set):

        cur = copy.deepcopy(inter)
        for pos, o in zip(abs_pos, absents):
            cur[pos].add(o)
        while set() in cur:
            cur.remove(set())

        f = phi_(cur, N_objs, G)
        X_inter_phi += f  # / np.linalg.norm(f)

    # weighting more the certain partitions
    # w = 1 / N_objs
    # w1 = 2 * w
    # w2 = round((1-w1*len(A_inter)),2) / len(absents)
    # for i in range(N_objs):
    #   X_inter_phi[i] = X_inter_phi[i]  * (w1 if i in A_inter else w2)

    return X_inter_phi / div


# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sushi_dataset import sushi_dataset as load_dataset

my_dpi = 96
N_objs = 10

X, X_ours, y, y_train, y_test, train_ind, test_ind, G =\
  load_dataset(0, "full", N=2500)

plt.figure(figsize=(500/my_dpi/1.6, 500/my_dpi/1.6), dpi=my_dpi)
# plt.figure()

# plt.subplot(1,2,1)

# A_1 = [set([x]) for x in X[2][:5]]+   [set([x for x in X[2][5:]])]
# A_2 = [set([x]) for x in X[2][:4:-1]]+[set([x for x in X[2][:5]])]

A_1 = X_ours[2]
A_2 = X_ours[2][::-1]

print(A_1)
print(A_2)

w_1 = phi_(A_1, N_objs, G)
w_2 = phi_(A_2, N_objs, G)

# plt.scatter(range(1,1+N_objs), w_1, label = r"$\phi(A_1)$")
# plt.scatter(range(1,1+N_objs), w_2, label = r"$\phi(A_2)$")
# plt.xticks(range(1,1+N_objs))
# plt.yticks(np.arange(-5,7,2))
plt.vlines(0,-0.5,10.5, "black", "--", alpha = 0.7)
plt.scatter(w_1, range(1,1+N_objs), label = r"$\phi(A)$", color = "gold")
plt.scatter(w_2, range(1,1+N_objs), label = r"$\phi(A')$", color = "red")
# plt.yticks(range(1,1+N_objs), df_sushi.iloc[:,0].tolist())
# plt.yticks(range(1,1+N_objs), [str(set([x])) for x in range(1,1+N_objs)])
plt.yticks(range(1,1+N_objs), range(1,1+N_objs))
# plt.xticks(np.arange(-5,7,2))
# plt.ylabel("W")
plt.xlabel(r"$\phi_d$", fontsize = 18)
plt.ylabel(r"$d$  ", fontsize = 18).set_rotation(0)
# plt.xlim(-6,8)
plt.ylim(0.5,10.5)
plt.legend(fontsize = 12, borderpad=0.01, borderaxespad=0, labelspacing=0.4,handletextpad=-0.3, scatterpoints=1, loc = "upper right")
plt.tight_layout()
plt.savefig("cached_results/interpretation21.pdf",  bbox_inches="tight")
plt.show()

# plt.subplot(1,2,2)
plt.figure(figsize=(500/my_dpi/1.6, 500/my_dpi/1.6), dpi=my_dpi)

plt.vlines(0,-0.5,9.5, "black", "--", alpha = 0.7)
plt.scatter(w_1*w_2, range(1,1+N_objs), color = "orange")#, label = r"$\phi(A_1)_i\cdot\phi(A_2)_i \forall i=1\ldots n$")
plt.yticks([])
plt.xlabel(r"$\phi(A)_d\phi(A')_d$", fontsize = 18)
# plt.xlim(-6,12)
plt.ylim(-0.5,9.5)

plt.tight_layout()
plt.savefig("cached_results/interpretation22.pdf",bbox_inches="tight")
# plt.legend()

plt.show()

# # %%


# str(w_1.round(2).tolist())[1:-1].replace(",", "\\")
# str(w_2.round(2).tolist())[1:-1].replace(",", "\\")

# str((w_1 * w_2).round(2).tolist())[1:-1].replace(",", "\\")


# # %%
from copy import deepcopy

A_is = deepcopy(X_ours[2])
v = []
k = len(A_is)
while len(v) < len(A_is):

    B_i  = reduce(lambda a,b: a.union(b), A_is[k-1:])
    B_i_ = reduce(lambda a,b: a.union(b), A_is[k:]) if k < len(A_is) else set([])
    v_ =  - (F(B_i, G) - F(B_i_, G)) / (len(B_i)-len(B_i_))

    if len(v) != 0 and v_ < v[0]:
        A_is[k-1:k+1] = [A_is[k-1].union(A_is[k])]
        v.pop(0)
        continue

    v.insert(0,v_)
    k -= 1

my_dpi = 96
plt.figure(figsize=(450/my_dpi/1.6, 250/my_dpi/1.6), dpi=my_dpi)

for i in range(len(A_is)):
  A_is[i] = set([x+1 for x in A_is[i]])
plt.hlines(0,-0.5,9.5, "black", "--", alpha = 0.7)
plt.scatter(range(len(v)) , -np.array(v), 10)
a = str(A_is).replace("[","").replace("]","").split("{")[1:]
a = [("\n" if i % 2 == 0 else "")+"{"+x.replace(",","") for i,x in enumerate(a) ]
plt.xticks(range(len(v)), a, fontsize = 7)
plt.yticks(np.arange(-4,6,2), fontsize = 7)
plt.yticks(fontsize = 7)
plt.xlim(-0.5,6.5)
plt.ylabel("values", fontsize = 8)


plt.tight_layout()
plt.savefig("cached_results/interpretation1.pdf")
plt.show()
