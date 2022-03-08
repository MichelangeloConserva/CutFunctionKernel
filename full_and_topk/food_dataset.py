import pandas as pd
import numpy as np
import math

from utils import create_graph

def synthetic_rank(pref, df_food, rng, sigma):
    importance = (5*np.array(pref)+1.)**-1

    vals = np.zeros(len(df_food.T))
    for i, indicator in enumerate(df_food.index):
        vals += importance[i] * df_food.loc[indicator,:].values
    vals += rng.randn(len(vals)) * sigma

    return np.argsort(vals)[::-1]  #  ranking[0] > ... > ranking[-1]


def food_dataset(seed, sigma, rank_type = "full", N = 100, prop_train = 0.8,
                  topk = None, min_removed = None, max_removed = None):
  N_objs = 8

  rng = np.random.RandomState(seed)

  data = [[9,7,10,0,2,1,4,4],
          [0,1,0,8,8,10,7,9],
          [3,0,7,8,9,10,7,6]]

  df_food = pd.DataFrame(data,
               columns = ["cake","biscuit","gelato","steak","burger","sausage",
                          "pasta","pizza"],
               index = ["sweet","savoury","juicy"]) / 10

  groups_ind = [[0,1,2],    # Sweet
                [3,4,5],    # Meat
                [6,7]]      # Carbs

  df_ranks = pd.DataFrame(columns = range(len(df_food.T)), index = range(N))

  # Sampling  sweet > savoury > juicy
  pref = [0,1,2]
  for i in range(N//2): df_ranks.iloc[i] = synthetic_rank(pref, df_food, rng, sigma)

  # Sampling  juicy > savoury > sweet
  pref = [2,1,0]
  for i in range(N//2,N): df_ranks.iloc[i] = synthetic_rank(pref, df_food, rng, sigma)

  mixing_ind = rng.choice(range(N), N, False)

  train_ind = np.arange(int(math.floor(prop_train*N))).reshape(-1,1)
  test_ind  = np.arange(int(math.floor(prop_train*N)),N).reshape(-1,1)

  X = df_ranks.values.astype(int)[mixing_ind]
  y = np.vstack(([0]*(N//2) + [1]*(N//2)))[mixing_ind]

  G, _ = create_graph(df_food.T, plot = False)


  if rank_type == "full":
    X_ours = np.empty(len(X), np.object)
    for i in range(len(X)):
      cur = []
      for s in X[i]: cur.append(set([s]))
      X_ours[i] = cur


  elif rank_type == "topk":
    assert not topk is None, "specify a value for topk"

    X = X[:, :topk]
    X_ours = np.empty(len(X), np.object)
    for i in range(len(X)):
        cur = []
        for s in X[i]:
            cur.append(set([s]))
        if len(cur) != N_objs: cur.append(set(range(N_objs)) - set(X[i]))
        X_ours[i] = cur


  elif rank_type == "exhaustive_interleaving":
    assert not min_removed is None and not max_removed is None, "specify a value for topk"

    X_ours = np.empty(len(X), np.object)
    for i in range(len(X)):
      cur = []
      for s in X[i]: cur.append(set([s]))

      n_removed = rng.randint(min_removed,max_removed)
      for _ in range(n_removed):
        iii = rng.choice(range(len(cur)-1), None, False)
        cur[iii] = cur[iii].union(cur[iii+1])
        cur.pop(iii+1)
      X_ours[i] = cur
    X = X_ours

  else: raise ValueError("The rank_type is not valid")


  return X, X_ours, y, y[train_ind.ravel()], y[test_ind.ravel()], train_ind, test_ind, G














































