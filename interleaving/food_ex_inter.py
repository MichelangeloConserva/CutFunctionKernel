# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.dummy import DummyClassifier

from our_kernel import get_phi
from kendall_kernel import KendallInterLeaving, KendallPartialflow
from gpflow_utils import fit_VGP, invlink
from food_dataset import food_dataset

np.set_printoptions(precision=2, suppress=True)
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
f64 = gpflow.utilities.to_default_float


def f1_score_test(m):
    pred_test = (m.predict_y(X_test)[0].numpy().squeeze() > 0.5).astype(int)
    return f1_score(y_test, pred_test)


seeds = [4, 12, 34, 42, 1232, 2638]
sigmas = np.linspace(0.1, 3, 15)
N_objs = 8

min_removed, max_removed = 2, 4

#%%

Our_all = []
Kendall_all = []
Dummy_all = []

loop = tqdm(seeds)
for seed in loop:
    Kendall_final = []
    Our_final = []
    Dummy_final = []

    tf.random.set_seed(seed)
    for sigma in sigmas:
        X, X_ours, y, y_train, y_test, train_ind, test_ind, G = food_dataset(
            seed,
            sigma,
            "exhaustive_interleaving",
            min_removed=min_removed,
            max_removed=max_removed,
        )

        ### DUMMY ###
        Dummy_final.append(
            f1_score(y, DummyClassifier(strategy="uniform").fit(X, y).predict(X))
        )

        ### OURS ###
        X_train, X_test = get_phi(X_ours, G, N_objs, train_ind, test_ind)
        ls = np.median(pairwise_distances(X_train))
        kern = gpflow.kernels.Linear()
        m_ours, _, _ = fit_VGP(
            (X_train, y_train), kern=kern, test_data=(X_test, y_test)
        )
        Our_final.append(f1_score_test(m_ours))

        ### Kendall ###
        KK = KendallInterLeaving(N_objs, N_objs, variance=1)
        K = np.ones((len(X), len(X)))
        for lll in range(len(X)):
            for j in range(lll, len(X)):
                K[lll, j] = K[j, lll] = KK.kernel_kendall(X[lll], X[j])

        X_train, X_test = train_ind.astype(float), test_ind.astype(float)
        kern = KendallPartialflow(K, N_objs)
        m_kend, f1scores, elbos = fit_VGP(
            (X_train, y_train), kern=kern, test_data=(X_test, y_test)
        )
        # Kendall_elbos.append(elbos); Kendall_f1train.append(f1scores)
        Kendall_final.append(f1_score_test(m_kend))
        loop.desc = str(seed)
        loop.refresh()

    Our_all.append(Our_final)
    Kendall_all.append(Kendall_final)
    Dummy_all.append(Dummy_final)


Our_all = np.array(Our_all)
Kendall_all = np.array(Kendall_all)
Dummy_all = np.array(Dummy_all)


# %%

# np.save("cached_results/Our_inter_food.npy", np.array(Our_all))
# np.save("cached_results/Kendall_inter_food.npy", np.array(Kendall_all))
# np.save("cached_results/Dummy_inter.npy", np.array(Dummy_all))

# %%

Our_all = np.load("cached_results/Our_inter_food.npy")
Kendall_all = np.load("cached_results/Kendall_inter_food.npy")
Dummy_all = np.load("cached_results/Dummy_inter.npy")
sigmas = np.linspace(0.1, 3, 15)
# sigmas = np.log(sigmas)

# %%
import seaborn as sns

sns.set_theme()

plus = 3
my_dpi = 96
markersize = 6
plt.figure(figsize=(400 / my_dpi, 280 / my_dpi), dpi=my_dpi)

plt.plot(
    sigmas,
    Kendall_all.mean(0),
    "-X",
    markersize=markersize,
    label="Kendall",
    color="tab:blue",
)
plt.fill_between(
    sigmas,
    Kendall_all.mean(0) + Kendall_all.std(0) / np.sqrt(len(Kendall_all)),
    Kendall_all.mean(0) - Kendall_all.std(0) / np.sqrt(len(Kendall_all)),
    color="tab:blue",
    alpha=0.2,
)


plt.plot(
    sigmas, Dummy_all.mean(0), "-*", markersize=markersize, label="Dummy", color="black"
)
plt.fill_between(
    sigmas,
    Dummy_all.mean(0) + Dummy_all.std(0) / np.sqrt(len(Kendall_all)),
    Dummy_all.mean(0) - Dummy_all.std(0) / np.sqrt(len(Kendall_all)),
    color="grey",
    alpha=0.2,
)

plt.plot(
    sigmas,
    Our_all.mean(0),
    "-o",
    markersize=markersize,
    label="Cut (Ours)",
    color="tab:orange",
)
plt.fill_between(
    sigmas,
    Our_all.mean(0) + Our_all.std(0) / np.sqrt(len(Kendall_all)),
    Our_all.mean(0) - Our_all.std(0) / np.sqrt(len(Kendall_all)),
    color="tab:orange",
    alpha=0.2,
)

# plt.title("GP classifier performance on test set", fontsize = 9+plus)
plt.xlabel(r"$\sigma$", fontsize=8 + plus)
plt.ylabel("F1-scores", fontsize=8 + plus)
# plt.legend(fontsize=7+plus, loc = 0)
plt.tight_layout()
plt.xticks([0.1] + np.arange(0.5, 3.5, 0.5).tolist(), fontsize=7 + plus)
plt.yticks(np.arange(0.5, 1.07, 0.1), fontsize=7 + plus)
plt.savefig("cached_results/food_finall_exinterleaving.pdf", bbox_inches="tight")
plt.show()
