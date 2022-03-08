import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.dummy import DummyClassifier

from our_kernel import get_phi
from mallows_kernel import Mallows_PRMCflow, get_phi_topk
from kendall_kernel import KendallPartialflow
from gpflow_utils import fit_VGP, invlink
from sushi_dataset import sushi_dataset as load_dataset

def f1_score_test(m):
    pred_test = (m.predict_y(X_test)[0].numpy().squeeze() > 0.5).astype(int)
    return f1_score(y_test, pred_test)


seeds = [4, 12, 34, 42, 1232, 2638]
topk = 6

# %% Precompute the Gram matrix for the Kendall kernel

# X, X_ours, y, y_train, y_test, train_ind, test_ind, G  =\
#   load_dataset(0, "topk", topk=topk, N=3000)

# from kendall_kernel import KendallTOPk
# KK = KendallTOPk(topk, 10, variance=1)
# K = np.ones((len(X),len(X)))
# for lll in trange(len(X)):
#     for j in range(lll,len(X)):
#         K[lll,j] = K[j,lll] = KK.kernel_kendall(X[lll],X[j])

# np.save(f"cached_results/top{topk}_kendall_gram.npy", K)


# %% VI

Dummy_final = []

Mallows_elbos = []
Mallows_final = []
Mallows_f1train = []

Kendall_elbos = []
Kendall_final = []
Kendall_f1train = []


Our_final = []
Our_f1train = []
Our_elbos = []


K_kendall = np.load(f"cached_results/top{topk}_kendall_gram.npy")


# %%

loop = tqdm(seeds, desc = "Our")
for seed in loop:

  X, X_ours, y, y_train, y_test, train_ind, test_ind, G  =\
    load_dataset(seed, "topk", topk=topk, N=2500)

  N_sushi = N_objs = 10
  N = len(X)
  tf.random.set_seed(seed)


  ### DUMMY ###
  dummy = DummyClassifier(strategy='uniform')
  Dummy_final.append(f1_score(y, dummy.fit(X, y).predict(X)))

  ### OURS ###
  X_train, X_test = get_phi(X_ours, G, N_objs, train_ind, test_ind)
  ls = np.median(pairwise_distances(X_train))

  kern =  gpflow.kernels.Linear() + gpflow.kernels.Constant()
  m_ours, f1scores, elbos = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))

  Our_elbos.append(elbos); Our_f1train.append(f1scores)
  Our_final.append(f1_score_test(m_ours))


  ### Mallows ###
  loop.desc = f"Mallows, Our f1={Our_final[-1]:.2f}"
  loop.refresh()
  n_mc_samples = 4
  X_train, X_test = get_phi_topk(X, n_mc_samples, N_objs, train_ind, test_ind)
  ls = np.median(pairwise_distances(X_train, None, metric = "hamming"))

  kern = Mallows_PRMCflow(ls = ls, n_mc_samples=n_mc_samples)
  m_Mall, f1scores, elbos = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
  Mallows_elbos.append(elbos); Mallows_f1train.append(f1scores)
  Mallows_final.append(f1_score_test(m_Mall))


  ### Kendall ###
  loop.desc = f"Kendall, Our={Our_final[-1]:.2f}, Mallows_final={Our_final[-1]:.2f}"
  loop.refresh()
  X_train, X_test = train_ind.astype(float), test_ind.astype(float)
  K_cur = K_kendall[np.vstack((train_ind, test_ind)).ravel()][:,np.vstack((train_ind, test_ind)).ravel()]

  kern = KendallPartialflow(K_cur, topk, N_objs)

  m_kend, f1scores, elbos = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
  Kendall_elbos.append(elbos); Kendall_f1train.append(f1scores)
  Kendall_final.append(f1_score_test(m_kend))




# %%
import pickle

# mem = [
# Kendall_final,
# Mallows_final ,
# Our_final,
# Dummy_final,
# Our_elbos,
# Mallows_elbos,
# Kendall_elbos,
# Our_f1train,
# Mallows_f1train,
# Kendall_f1train
# ]
#
# with open(f'cached_results/top{topk}_sushi.pkl', 'wb') as f:
#     pickle.dump(mem, f)

# with open(f'cached_results/top{topk}_sushi.pkl', "rb") as f:
#     Kendall_final,\
#     Mallows_final ,\
#     Our_final,\
#     Dummy_final,\
#     Our_elbos,\
#     Mallows_elbos,\
#     Kendall_elbos,\
#     Our_f1train,\
#     Mallows_f1train,\
#     Kendall_f1train = pickle.load(f)


# %%

import seaborn as sns
sns.set_theme()

my_dpi = 96

plt.figure(figsize=(700/my_dpi/1.6, 300/my_dpi/1.6), dpi=my_dpi)

plt.subplot(1,2,1)

mmax = 0

for e in Our_elbos:
  o = plt.plot(np.log2(range(1,1+len(e))), e, color = "tab:orange")[0]
  if len(e) > mmax: mmax = len(e)

for e in Kendall_elbos:
  k = plt.plot(np.log2(range(1,1+len(e))), e, color = "tab:olive")[0]
  if len(e) > mmax: mmax = len(e)

for e in Mallows_elbos:
  m = plt.plot(np.log2(range(1,1+len(e))), e, color = "tab:blue")[0]
  if len(e) > mmax: mmax = len(e)


plt.xticks(np.log2([1,2,4,8,16,32,64]), [1,2,4,8,16,32,64],
            fontsize=7)
plt.yticks(np.arange(-4500, - 1000, 500), fontsize=7)
plt.legend([o,k,m], ["Cut (Ours)", "Kendall", "Mallows"], fontsize = 7)
plt.xlabel("Iteration", fontsize = 8)
plt.title("Elbo", fontsize = 9)


# ax = plt.subplot(1,3,2)
# for e in Kendall_f1train: k = plt.plot(e, color = "tab:olive")[0]
# for e in Mallows_f1train: m = plt.plot(e, color = "tab:blue")[0]
# for e in Our_f1train:     o = plt.plot(e, color = "tab:orange")[0]

# plt.xticks(range(0, mmax, 15), fontsize=8)
# plt.yticks(fontsize=8)
# # plt.legend([o,k,m], ["Cut (Ours)", "Kendall", "Mallows"])
# plt.xlabel("Iteration", fontsize = 11)
# plt.title("F1-score Train")
# plt.ylim(0.2,0.88)


ax = plt.subplot(1,2,2)

b_plt = np.vstack((Kendall_final, Our_final, Mallows_final)).T

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
parts = plt.violinplot(b_plt,showmeans=False, showmedians=False,
          showextrema=False, bw_method=0.55)

for pc,c in zip(parts['bodies'],["tab:olive","tab:orange","tab:blue"]):
    pc.set_facecolor(c)
    pc.set_edgecolor('black')
    pc.set_alpha(1)


quartile1, medians, quartile3 = np.percentile(b_plt, [25, 50, 75], axis=0)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(b_plt.T, quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
plt.vlines(inds, quartile1, quartile3, color=["lawngreen","yellow","cyan"], linestyle='-', lw=1)
# plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
# plt.scatter(inds, medians, marker='o', color='white', s=7, zorder=2)
plt.scatter(inds, medians, marker='o', color=["lawngreen","yellow","cyan"], s=7, zorder=2)



plt.hlines(np.mean(Dummy_final), -1, 40, "tab:blue", "-.", label = "Dummy")
plt.xlim(0.5,3.5)
# plt.ylim(0,1)
plt.title("F1-score", fontsize=9)

plt.xticks(range(1,4), ["Kendall","\nCut (Ours)", "Mallows"],
            fontsize = 7)

plt.yticks(np.arange(0.4,0.65, 0.05), fontsize=7)

# plt.legend(fontsize=8, loc = 3)#, loc = 0)


plt.tight_layout()
plt.savefig(f"cached_results/sushi_top{topk}_final.pdf", bbox_inches="tight")
plt.show()