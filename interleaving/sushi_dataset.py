# %% Imports
import numpy as np
import math
import pandas as pd

from utils import create_graph


def load_sushi_dataset(
    DATA_PATH_SUSHI="./data/sushi/sushi3.idata",
    DATA_PATH_RANK="./data/sushi/sushi3a.5000.10.order",
    DATA_PATH_USER="./data/sushi/sushi3.udata",
    minor_group=False,
):

    df_sushi = pd.read_csv(DATA_PATH_SUSHI, sep="\t", header=None, index_col=0)
    df_sushi.columns = [
        "name",
        "style",
        "Major group",
        "Minor group",
        "Oiliness",
        "Eat frequency",
        "Price",
        "Sold frequency",
    ]

    if minor_group:
        df_sushi = pd.get_dummies(df_sushi, columns=["Minor group"], dtype=int)
    else:
        df_sushi.drop("Minor group", axis=1, inplace=True)

    sub_indices_SET_A = [0, 1, 2, 3, 4, 6, 7, 8, 26, 29]
    df_sushi = df_sushi.iloc[sub_indices_SET_A]

    df_ranking = pd.read_csv(DATA_PATH_RANK, sep=" ", skiprows=1).iloc[:, 2:]

    # Reading datasets
    age = np.loadtxt(DATA_PATH_USER)[:, 2]  # column for the age
    region = np.loadtxt(DATA_PATH_USER)[:, 8]  # column for the region ID
    y = np.array([0 if x <= 5 else 1 for x in region]).reshape(-1, 1)

    X = np.loadtxt(DATA_PATH_RANK, delimiter=" ", skiprows=1).astype(int)[:, 2:]
    df_ranking = pd.DataFrame(X)

    # Reading datasets
    # region = np.loadtxt(DATA_PATH_USER)
    # y = region[:,6].reshape(-1,1)

    # X = np.loadtxt(DATA_PATH_RANK, delimiter = " ").astype(int)[:,2:]
    # df_ranking = pd.DataFrame(X)

    return df_sushi, df_ranking, X, y


def sushi_dataset(
    seed,
    rank_type="full",
    N=100,
    prop_train=0.8,
    topk=None,
    min_removed=None,
    max_removed=None,
    interlength=None,
    score_par=1,
    approximate=None,
):
    N_sushi = 10

    rsg = np.random.RandomState(seed)
    df_objs, df_ranking, X, y = load_sushi_dataset()

    g_data = df_objs.iloc[:, 1:]
    g_data = g_data / g_data.max(axis=0).values  # Normalize

    G, similarities = create_graph(
        g_data, plot=False, score_par=score_par, approximate=approximate
    )

    if N < 3100:
        idxs_subsample = np.arange(len(X))[y.ravel() == 1]
        idxs_subsample = np.hstack(
            (
                idxs_subsample,
                rsg.choice(
                    np.arange(len(X))[y.ravel() == 0], len(idxs_subsample), False
                ),
            )
        )
        rsg.shuffle(idxs_subsample)
        idxs_subsample = idxs_subsample[:N]
    else:
        idxs_subsample = rsg.choice(np.arange(len(df_ranking)), N, False)

    X = df_ranking.values[idxs_subsample, :].copy().astype(int)
    y = y[idxs_subsample, :].copy().astype(int)

    train_ind = np.arange(int(math.floor(prop_train * N))).reshape(-1, 1)
    test_ind = np.arange(int(math.floor(prop_train * N)), len(X)).reshape(-1, 1)

    if rank_type == "full":
        X_ours = np.empty(len(X), np.object)
        for i in range(len(X)):
            cur = []
            for s in X[i]:
                cur.append(set([s]))
            X_ours[i] = cur

    elif rank_type == "topk":
        assert not topk is None, "specify a value for topk"

        X_ours = np.empty(len(X), np.object)
        for i in range(len(X)):
            cur = []
            for s in X[i]:
                cur.append(set([s]))
                if len(cur) == topk:
                    break
            if len(cur) != N_sushi:
                cur.append(set(range(N_sushi)) - set(X[i, :topk]))
            X_ours[i] = cur
        X = X[:, :topk]

    elif rank_type == "exhaustive_interleaving":
        assert (
            not min_removed is None and not max_removed is None
        ), "specify a value for topk"

        X_ours = np.empty(len(X), np.object)
        for i in range(len(X)):
            cur = []
            for s in X[i]:
                cur.append(set([s]))

            n_removed = rsg.randint(min_removed, max_removed)
            for _ in range(n_removed):
                iii = rsg.choice(range(len(cur) - 1), None, False)
                cur[iii] = cur[iii].union(cur[iii + 1])
                cur.pop(iii + 1)
            X_ours[i] = cur
        X = X_ours

    elif rank_type == "interleaving":

        X_ours = np.empty(len(X), np.object)
        for i in range(len(X)):
            X_ours[i] = list(
                np.array(X[i])[
                    sorted(np.random.choice(range(N_sushi), interlength, False))
                ]
            )
        # X = np.vstack(X)
        X = X_ours = X_ours.tolist()

    else:
        raise ValueError("The rank_type is not valid")

    return (
        X,
        X_ours,
        y,
        y[train_ind.ravel()],
        y[test_ind.ravel()],
        train_ind,
        test_ind,
        G,
    )


def sushi_large(
    seed,
    N,
    prop_train=0.8,
    DATA_PATH_SUSHI="./data/sushi/sushi3.idata",
    DATA_PATH_RANK="./data/sushi/sushi3b.5000.10.order",
    DATA_PATH_USER="./data/sushi/sushi3.udata",
    minor_group=False,
    approximate=0.4,
):

    df_sushi = pd.read_csv(DATA_PATH_SUSHI, sep="\t", header=None, index_col=0)
    df_sushi.columns = [
        "name",
        "style",
        "Major group",
        "Minor group",
        "Oiliness",
        "Eat frequency",
        "Price",
        "Sold frequency",
    ]

    if minor_group:
        df_sushi = pd.get_dummies(df_sushi, columns=["Minor group"], dtype=int)
    else:
        df_sushi.drop("Minor group", axis=1, inplace=True)

    df_ranking = pd.read_csv(DATA_PATH_RANK, sep=" ", skiprows=1).iloc[:, 2:]

    # Reading datasets
    region = np.loadtxt(DATA_PATH_USER)[:, 8]  # column for the region ID
    y = np.array([0 if x <= 5 else 1 for x in region]).reshape(-1, 1)

    X = np.loadtxt(DATA_PATH_RANK, delimiter=" ", skiprows=1).astype(int)[:, 2:]
    df_ranking = pd.DataFrame(X)

    rsg = np.random.RandomState(seed)

    g_data = df_sushi.iloc[:, 1:]
    g_data = g_data / g_data.max(axis=0).values  # Normalize

    G, similarities = create_graph(g_data, plot=False, approximate=approximate)

    if N < 3100:
        idxs_subsample = np.arange(len(X))[y.ravel() == 1]
        idxs_subsample = np.hstack(
            (
                idxs_subsample,
                rsg.choice(
                    np.arange(len(X))[y.ravel() == 0], len(idxs_subsample), False
                ),
            )
        )
        rsg.shuffle(idxs_subsample)
        idxs_subsample = idxs_subsample[:N]
    else:
        idxs_subsample = rsg.choice(np.arange(len(df_ranking)), N, False)

    X = df_ranking.values[idxs_subsample, :].copy().astype(int)
    y = y[idxs_subsample, :].copy().astype(int)

    train_ind = np.arange(int(math.floor(prop_train * N))).reshape(-1, 1)
    test_ind = np.arange(int(math.floor(prop_train * N)), len(X)).reshape(-1, 1)

    return (
        X.tolist(),
        y,
        y[train_ind.ravel()],
        y[test_ind.ravel()],
        train_ind,
        test_ind,
        G,
    )
