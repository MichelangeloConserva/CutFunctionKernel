# %% Imports
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
from kendall_kernel import KendallTOPk, KendallPartialflow
from gpflow_utils import fit_VGP, invlink
from food_dataset import food_dataset as load_dataset

def f1_score_test(m):
    pred_test = (m.predict_y(X_test)[0].numpy().squeeze() > 0.5).astype(int)
    return f1_score(y_test, pred_test)

seeds = [4, 12, 34, 42, 1232, 2638]
sigmas = np.linspace(0.1,3,15)
N_objs = 8
N = 250
topk = 6


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
    X, X_ours, y, y_train, y_test, train_ind, test_ind, G =\
      load_dataset(seed, sigma, "topk", topk=topk, N=N)

    ### DUMMY ###
    Dummy_final.append(f1_score(y, DummyClassifier(strategy='uniform').fit(X, y).predict(X)))

    ### OURS ###
    X_train, X_test = get_phi(X_ours, G, N_objs, train_ind, test_ind)
    ls = np.median(pairwise_distances(X_train))
    # kern =  gpflow.kernels.Linear() * gpflow.kernels.Matern32(lengthscales=1/ls) + gpflow.kernels.Constant()
    kern =  gpflow.kernels.Linear() + gpflow.kernels.Constant()
    m_ours, _, _ = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
    Our_final.append(f1_score_test(m_ours))

    ### Mallows ###
    # n_mc_samples = 4
    # X_train, X_test = get_phi_topk(X, n_mc_samples, N_objs, train_ind, test_ind)
    # ls = np.median(pairwise_distances(X_train, None, metric = "hamming"))
    #
    # kern = Mallows_PRMCflow(ls = ls, n_mc_samples=n_mc_samples)
    # m_Mall, _, _ = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
    # Mallows_final.append(f1_score_test(m_Mall))
    #
    # ### Kendall ###
    # KK = KendallTOPk(topk, 10, variance=1)
    # K = np.ones((len(X),len(X)))
    # for lll in range(len(X)):
    #     for j in range(lll,len(X)):
    #         K[lll,j] = K[j,lll] = KK.kernel_kendall(X[lll],X[j])
    # X_train, X_test = train_ind.astype(float), test_ind.astype(float)
    # kern = KendallPartialflow(K, N_objs)
    # m_kend, _, _ = fit_VGP((X_train, y_train), kern = kern, test_data=(X_test,y_test))
    # Kendall_final.append(f1_score_test(m_kend))

  Our_all.append(Our_final)
  Mallows_all.append(Mallows_final)
  Kendall_all.append(Kendall_final)
  Dummy_all.append(Dummy_final)

Our_all = np.array(Our_all)
Mallows_all = np.array(Mallows_all)
Kendall_all = np.array(Kendall_all)
Dummy_all = np.array(Dummy_all)


# %%
np.save(f"cached_results/Our_alltop{topk}.npy", np.array(Our_all))
np.save(f"cached_results/Mallows_alltop{topk}.npy", np.array(Mallows_all))
np.save(f"cached_results/Kendall_alltop{topk}.npy", np.array(Kendall_all))
np.save(f"cached_results/Dummy_alltop{topk}.npy", np.array(Dummy_all))

# %%
Our_all = np.load(f"cached_results/Our_alltop{topk}.npy")
Mallows_all = np.load(f"cached_results/Mallows_alltop{topk}.npy")
Kendall_all = np.load(f"cached_results/Kendall_alltop{topk}.npy")
Dummy_all = np.load(f"cached_results/Dummy_alltop{topk}.npy")

# %%
import seaborn as sns
sns.set_theme()
from matplotlib import colors


if topk == 3:
    import seaborn as sns

    sns.set_theme()

    plus = 3

    my_dpi = 96
    markersize = 6
    plt.figure(figsize=(400 / my_dpi, 280 / my_dpi), dpi=my_dpi)

    plt.plot(sigmas, Kendall_all.mean(0), "-X", markersize=markersize,
             label="Kendall", color="tab:blue")
    plt.fill_between(sigmas,
                     Kendall_all.mean(0) + Kendall_all.std(0) / np.sqrt(len(Kendall_all)),
                     Kendall_all.mean(0) - Kendall_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:blue", alpha=0.2)

    plt.plot(sigmas, Mallows_all.mean(0), "-P", markersize=markersize,
             label="Mallows", color="tab:olive")
    plt.fill_between(sigmas,
                     Mallows_all.mean(0) + Mallows_all.std(0) / np.sqrt(len(Kendall_all)),
                     Mallows_all.mean(0) - Mallows_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:olive", alpha=0.2)

    plt.plot(sigmas, Dummy_all.mean(0), "-*", markersize=markersize, label="Dummy",
             color="black")
    plt.fill_between(sigmas,
                     Dummy_all.mean(0) + Dummy_all.std(0) / np.sqrt(len(Kendall_all)),
                     Dummy_all.mean(0) - Dummy_all.std(0) / np.sqrt(len(Kendall_all)), color="grey", alpha=0.2)

    plt.plot(sigmas, Our_all.mean(0), "-o", markersize=markersize,
             label="Cut (Ours)", color="tab:orange")
    plt.fill_between(sigmas,
                     Our_all.mean(0) + Our_all.std(0) / np.sqrt(len(Kendall_all)),
                     Our_all.mean(0) - Our_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:orange", alpha=0.2)

    # plt.title("GP classifier performance on test set", fontsize = 9+plus)
    plt.xlabel(r"$\sigma$", fontsize=8 + plus)
    plt.ylabel("F1-scores", fontsize=8 + plus)
    # plt.legend(fontsize=7+plus, loc = 0)
    plt.tight_layout()
    plt.xticks([0.1] + np.arange(0.5, 3.5, 0.5).tolist(), fontsize=7 + plus)
    plt.yticks(np.arange(0.5, 1.07, 0.1), fontsize=7 + plus)
    plt.savefig(f"cached_results/food_finall_top{topk}.pdf", bbox_inches="tight")
    plt.show()
else:

    my_dpi = 96
    markersize = 8
    plt.figure(figsize=(600/my_dpi, 300/my_dpi), dpi=my_dpi)

    plt.plot(sigmas, Kendall_all.mean(0), "-P", markersize = markersize,
              label = "Kendall" , color = "tab:blue")
    plt.fill_between(sigmas,
                      Kendall_all.mean(0) + Kendall_all.std(0) / np.sqrt(len(Kendall_all)),
                      Kendall_all.mean(0) - Kendall_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:blue", alpha = 0.2)

    plt.plot(sigmas, Mallows_all.mean(0), "-P", markersize=markersize,
             label="Mallows", color="tab:olive")
    plt.fill_between(sigmas,
                     Mallows_all.mean(0) + Mallows_all.std(0) / np.sqrt(len(Kendall_all)),
                     Mallows_all.mean(0) - Mallows_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:olive", alpha=0.2)

    plt.plot(sigmas, Dummy_all.mean(0), "-*", markersize = markersize,label = "Dummy",
              color = "black")
    plt.fill_between(sigmas,
                      Dummy_all.mean(0) + Dummy_all.std(0) / np.sqrt(len(Kendall_all)),
                      Dummy_all.mean(0) - Dummy_all.std(0) / np.sqrt(len(Kendall_all)), color="grey", alpha = 0.2)

    plt.plot(sigmas, Our_all.mean(0), "-o", markersize = markersize,
              label = "Our", color = "tab:orange")
    plt.fill_between(sigmas,
                      Our_all.mean(0) + Our_all.std(0) / np.sqrt(len(Kendall_all)),
                      Our_all.mean(0) - Our_all.std(0) / np.sqrt(len(Kendall_all)), color="tab:orange", alpha = 0.2)

    plt.xlabel(r"$\sigma$")
    plt.ylabel("F1-score")
    # plt.legend(fontsize=9, loc = 0)
    plt.tight_layout()
    plt.savefig(f"cached_results/food_finall_top{topk}.pdf")
    plt.show()

