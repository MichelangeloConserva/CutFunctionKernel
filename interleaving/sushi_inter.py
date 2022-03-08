import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm, trange
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.dummy import DummyClassifier

from our_kernel import phi_interleaving
from kendall_kernel import KendallInterLeavingNonEx, KendallPartialflow
from gpflow_utils import fit_VGP, invlink
from sushi_dataset import sushi_dataset as load_dataset


def f1_score_test(m):
    pred_test = (m.predict_y(X_test)[0].numpy().squeeze() > 0.5).astype(int)
    return f1_score(y_test, pred_test)


seeds = [4, 12, 34, 42, 1232, 2638]
interlength = 4
N_sushi = N_objs = 10

Dummy_final = []

Kendall_elbos = []
Kendall_final = []
Kendall_f1train = []

Our_final = []
Our_f1train = []
Our_elbos = []


# %%

if not os.path.isfile(f"cached_results/inter{interlength}_kendall_gram.npy"):
    X, _, y, y_train, y_test, train_ind, test_ind, G  =\
      load_dataset(0, N=3000, rank_type="interleaving", interlength = interlength)

    KK = KendallInterLeavingNonEx(6, N_objs, variance = 1)
    K = np.ones((len(X),len(X)))
    inputs = []
    for lll in range(len(X)):
        for j in range(lll,len(X)):
            inputs.append((lll, j, X[lll],X[j], KK))
    inputs = np.array_split(inputs, 14*3)

    def f(args):
        res = []
        for lll, j, X_lll, X_j, KK in args:
            x = KK.kernel_kendall(X_lll, X_j)
            res.append((lll, j, x))
        return res
    from multiprocessing import Pool

    with Pool(processes=14) as p:
        with tqdm(total=len(inputs)) as pbar:
            for r in p.imap_unordered(f, inputs):
                for lll, j, x in r:
                    K[lll, j] = K[j, lll] = x
                pbar.update()


    np.save(f"cached_results/inter{interlength}_kendall_gram.npy", K)

# %%

# The saved values are exact, no heuristic

# samples = 150

loop = tqdm(seeds, desc="Cut (Ours)")
for seed in loop:

    X, _, y, y_train, y_test, train_ind, test_ind, G = load_dataset(
        seed, N=2500, rank_type="interleaving", interlength=interlength
    )

    N = len(X)
    tf.random.set_seed(seed)
    dummy = DummyClassifier(strategy="uniform")
    Dummy_final.append(f1_score(y, dummy.fit(X, y).predict(X)))

    if os.path.isfile(f"cached_results/X_train_{seed}_inter.npy"):
        X_train = np.load(f"cached_results/X_train_{seed}_inter.npy")
        X_test = np.load(f"cached_results/X_test_{seed}_inter.npy")
    else:
        ### OURS ###
        def f(args):
            res = []
            for i, X_i, G, N_objs, b in args:
                res.append((i, phi_interleaving(X_i, G, N_objs, b)))
            return res

        inputs = []
        for i in train_ind.T.tolist()[0]:
            inputs.append((i, X[i], G, N_objs, False))
        inputs = np.array_split(inputs, 14*3)

        X_train = np.empty((int(len(X) * 0.8), N_objs))
        with Pool(processes=14) as p:
            for r in p.imap_unordered(f, inputs):
                for i, phi in r:
                    X_train[i] = phi

        X_test = np.empty((len(test_ind), N_objs))
        inputs = []
        for j, i in enumerate(test_ind.T.tolist()[0]):
            inputs.append((j, X[i], G, N_objs, False))
        inputs = np.array_split(inputs, 14*3)

        with Pool(processes=14) as p:
            for r in p.imap_unordered(f, inputs):
                for i, phi in r:
                    X_test[i] = phi

        np.save(f"cached_results/X_train_{seed}_inter", X_train)
        np.save(f"cached_results/X_test_{seed}_inter", X_test)

    kern = gpflow.kernels.Linear()
    m_ours, f1scores, elbos = fit_VGP(
        (X_train, y_train), kern=kern, test_data=(X_test, y_test), max_iter=10
    )
    Our_elbos.append(elbos)
    Our_f1train.append(f1scores)
    Our_final.append(f1_score_test(m_ours))
    print(Our_final)

# %%
import seaborn as sns
sns.set_theme()


X, _, y, y_train, y_test, train_ind, test_ind, G = load_dataset(
    0, N=3000, rank_type="interleaving", interlength=interlength
)
K = np.load(f"cached_results/inter{interlength}_kendall_gram.npy")

loop = tqdm(seeds, desc="Kendall")
for seed in loop:

    rng = np.random.RandomState(seed)
    ind = rng.choice(np.arange(len(y)), 2500, False).astype(int)
    y = y[ind]
    train_ind, test_ind = np.arange(2000).reshape(-1, 1), np.arange(2000, 2500).reshape(
        -1, 1
    )
    y_train, y_test = y[train_ind.ravel()], y[test_ind.ravel()]

    ### Kendall ###
    KK = KendallInterLeavingNonEx(6, N_objs, variance=1)
    K_cur = K[ind, :][:, ind]
    X_train, X_test = train_ind.astype(float), test_ind.astype(float)
    kern = KendallPartialflow(K_cur, N_objs)
    m_kend, f1scores, elbos = fit_VGP(
        (X_train, y_train), kern=kern, test_data=(X_test, y_test)
    )
    Kendall_elbos.append(elbos)
    Kendall_f1train.append(f1scores)
    Kendall_final.append(f1_score_test(m_kend))
    print(Kendall_final)


# %%
import pickle

mem = [
    Kendall_final,
    Our_final,
    Dummy_final,
    Our_elbos,
    Kendall_elbos,
    Our_f1train,
    Kendall_f1train,
]

with open(f"cached_results/inter{interlength}_sushi.pkl", "wb") as f:
    pickle.dump(mem, f)

with open(f"cached_results/inter{interlength}_sushi.pkl", "rb") as f:
    (
        Kendall_final,
        Our_final,
        Dummy_final,
        Our_elbos,
        Kendall_elbos,
        Our_f1train,
        Kendall_f1train,
    ) = pickle.load(f)


# %%

import seaborn as sns
sns.set_theme()

my_dpi = 96

plt.figure(figsize=(700 / my_dpi / 1.6, 300 / my_dpi / 1.6), dpi=my_dpi)

plt.subplot(1, 2, 1)

mmax = 0
for e in Our_elbos:
    # o = plt.plot(e, color = "tab:orange")[0]
    o = plt.plot(np.log2(range(1, 1 + len(e))), e, color="tab:orange")[0]
    if len(e) > mmax:
        mmax = len(e)

for e in Kendall_elbos:
    # k = plt.plot(e, color = "tab:olive")[0]
    k = plt.plot(np.log2(range(1, 1 + len(e))), e, color="tab:olive")[0]
    if len(e) > mmax:
        mmax = len(e)

plt.xticks(np.log2([1, 2, 4, 8, 16, 32]), [1, 2, 4, 8, 16, 32], fontsize=7)
# plt.xticks(np.linspace(0, np.log2(mmax), 11),
#             np.linspace(0, mmax, 11).astype(int),
#             fontsize=7)
plt.yticks(fontsize=8)
plt.legend([o, k], ["Cut (Ours)", "Kendall"], fontsize=7)
plt.xlabel("Iteration", fontsize=8)
plt.title("Elbo", fontsize=9)


# ax = plt.subplot(1,3,2)
# for e in Our_f1train:     o = plt.plot(e, color = "tab:orange")[0]
# for e in Kendall_f1train: k = plt.plot(e, color = "tab:olive")[0]

# plt.xticks(range(0, mmax, 10), fontsize=8)
# plt.yticks(fontsize=8)
# # plt.legend([o,k,m], ["Cut (Ours)", "Kendall", "Mallows"])
# plt.xlabel("Iteration", fontsize = 11)
# plt.title("F1-score Train")
# plt.ylim(0.2,0.88)


ax = plt.subplot(1, 2, 2)

b_plt = np.vstack((Kendall_final, Our_final)).T

positions = [1.25, 2.75]


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


parts = plt.violinplot(
    b_plt,
    positions=positions,
    showmeans=False,
    showmedians=False,
    showextrema=False,
    bw_method=0.55,
)

for pc, c in zip(parts["bodies"], ["tab:olive", "tab:orange"]):
    pc.set_facecolor(c)
    pc.set_edgecolor("black")
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(b_plt, [25, 50, 75], axis=0)
whiskers = np.array(
    [
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(b_plt.T, quartile1, quartile3)
    ]
)
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

inds = positions
plt.vlines(
    inds, quartile1, quartile3, color=["lawngreen", "yellow"], linestyle="-", lw=1
)
# plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
# plt.scatter(inds, medians, marker='o', color='white', s=7, zorder=2)
plt.scatter(inds, medians, marker="o", color=["lawngreen", "yellow"], s=7, zorder=2)


plt.hlines(np.mean(Dummy_final), -1, 40, "tab:blue", "-.", label="Dummy")
plt.xlim(0.5, 3.5)
# plt.ylim(0,1)
plt.title("F1-score", fontsize=9)

plt.xticks(positions, ["Kendall", "Ours"], fontsize=7)

plt.yticks(np.arange(0.4, 0.65, 0.05), fontsize=9)

# plt.legend(fontsize=8, loc = 3)#, loc = 0)


plt.tight_layout()
plt.savefig(f"cached_results/inter{interlength}_sushi.pdf")
plt.show()
