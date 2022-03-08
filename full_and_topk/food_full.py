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
from mallows_kernel import MallowsFlow, get_phi_mallows
from kendall_kernel import get_phi_kendall
from gpflow_utils import fit_VGP, invlink
from food_dataset import food_dataset as load_dataset


def f1_score_test(m):
    pred_test = (m.predict_y(X_test)[0].numpy().squeeze() > 0.5).astype(int)
    return f1_score(y_test, pred_test)


seeds = [4, 12, 34, 42, 1232, 2638]
sigmas = np.linspace(0.1, 3, 15)
N_objs = 8
N = 250


#%%

Our_all = []
Mallows_all = []
Kendall_all = []
Dummy_all = []

for seed in tqdm(seeds):
    Kendall_final = []
    Mallows_final = []
    Our_final = []
    Dummy_final = []

    tf.random.set_seed(seed)
    for sigma in sigmas:
        X, X_ours, y, y_train, y_test, train_ind, test_ind, G = load_dataset(
            seed, sigma, N=N
        )

        ### DUMMY ###
        Dummy_final.append(
            f1_score(y, DummyClassifier(strategy="uniform").fit(X, y).predict(X))
        )

        ### OURS ###
        X_train, X_test = get_phi(X_ours, G, N_objs, train_ind, test_ind)
        ls = np.median(pairwise_distances(X_train))
        kern = (
            gpflow.kernels.Linear() + gpflow.kernels.Constant()
        )  # * gpflow.kernels.Matern32(lengthscales=1/ls)
        m_ours, _, _ = fit_VGP(
            (X_train, y_train), kern=kern, test_data=(X_test, y_test)
        )
        Our_final.append(f1_score_test(m_ours))

        ### Mallows ###
        # X_train, X_test = get_phi_mallows(X, N_objs, train_ind, test_ind)
        # ls = np.median(pairwise_distances(X_train, None, metric = "hamming"))
        # kern =  MallowsFlow(1/ls)
        # m_Mall, _, _ = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
        # Mallows_final.append(f1_score_test(m_Mall))
        #
        # ### Kendall ###
        # X_train, X_test = get_phi_kendall(X, N_objs, train_ind, test_ind)
        # kern = gpflow.kernels.Linear()
        # m_Kend, _, _ = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
        # Kendall_final.append(f1_score_test(m_Kend))

    Our_all.append(Our_final)
    Mallows_all.append(Mallows_final)
    Kendall_all.append(Kendall_final)
    Dummy_all.append(Dummy_final)


Our_all = np.array(Our_all)
Mallows_all = np.array(Mallows_all)
Kendall_all = np.array(Kendall_all)
Dummy_all = np.array(Dummy_all)

Our_all = Our_all.tolist()
Mallows_all = Mallows_all.tolist()
Kendall_all = Kendall_all.tolist()
Dummy_all = Dummy_all.tolist()
#

# %%

np.save("cached_results/Our_all_food.npy", np.array(Our_all))
np.save("cached_results/Mallows_all_food.npy", np.array(Mallows_all))
np.save("cached_results/Kendall_all_food.npy", np.array(Kendall_all))
np.save("cached_results/Dummy_all_food.npy", np.array(Dummy_all))


# %%

Our_all = np.load("cached_results/Our_all_food.npy")
Mallows_all = np.load("cached_results/Mallows_all_food.npy")
Kendall_all = np.load("cached_results/Kendall_all_food.npy")
Dummy_all = np.load("cached_results/Dummy_all_food.npy")


# %%
import seaborn as sns

sns.set_theme()


my_dpi = 96
markersize = 8
plt.figure(figsize=(600 / my_dpi, 300 / my_dpi), dpi=my_dpi)

plt.plot(
    sigmas,
    Kendall_all.mean(0),
    "-P",
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
    sigmas,
    Mallows_all.mean(0),
    "-P",
    markersize=markersize,
    label="Mallows",
    color="tab:olive",
)
plt.fill_between(
    sigmas,
    Mallows_all.mean(0) + Mallows_all.std(0) / np.sqrt(len(Kendall_all)),
    Mallows_all.mean(0) - Mallows_all.std(0) / np.sqrt(len(Kendall_all)),
    color="tab:olive",
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
    label="Our",
    color="tab:orange",
)
plt.fill_between(
    sigmas,
    Our_all.mean(0) + Our_all.std(0) / np.sqrt(len(Kendall_all)),
    Our_all.mean(0) - Our_all.std(0) / np.sqrt(len(Kendall_all)),
    color="tab:orange",
    alpha=0.2,
)

plt.xlabel(r"$\sigma$")
# plt.title("F1-scores")
plt.ylabel("F1-score")
plt.legend(fontsize=9, loc=0)
plt.xticks([0.1] + np.arange(0.5, 3.5, 0.5).tolist())
plt.tight_layout()
plt.savefig("cached_results/food_finall_full.pdf", bbox_inches="tight")
plt.show()
