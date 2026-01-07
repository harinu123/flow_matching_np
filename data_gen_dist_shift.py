
import os
import numpy as np
import pandas as pd

np.random.seed(0)

def f(a,b,c,x):
    return a*x + np.sin(b*x) + c

out_dir = "data/trending-sinusoids-dist-shift"
os.makedirs(out_dir, exist_ok=True)

N = 2000
n_train, n_val, n_test = 1000, 500, 500
T = 100
x = np.linspace(-2, 2, T)

curves = []
knowledge = []
curve_ids = list(range(N))

# train + val (in-support)
for _ in range(n_train + n_val):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(0.0, 3.0)
    c = np.random.uniform(-1, 1)
    y = f(a,b,c,x)
    curves.append(y)
    knowledge.append((a,b,c))

# test (out-of-support, Rhis is a much harder shift)
for _ in range(n_test):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(4.0, 6.0)
    c = np.random.uniform(-1, 1)
    y = f(a,b,c,x)
    curves.append(y)
    knowledge.append((a,b,c))

curves_df = pd.DataFrame(np.stack(curves, axis=0))
curves_df["curve_id"] = curve_ids

knowledge_df = pd.DataFrame(knowledge, columns=["a","b","c"])
knowledge_df["curve_id"] = curve_ids

splits = ["train"]*n_train + ["val"]*n_val + ["test"]*n_test
split_df = pd.DataFrame({"curve_id": curve_ids, "split": splits})

curves_df.to_csv(f"{out_dir}/data.csv", index=False)
knowledge_df.to_csv(f"{out_dir}/knowledge.csv", index=False)
split_df.to_csv(f"{out_dir}/splits.csv", index=False)

print("Wrote:", out_dir)
print("Train/val b range:",
      knowledge_df.iloc[:n_train+n_val]["b"].min(),
      knowledge_df.iloc[:n_train+n_val]["b"].max())
print("Test b range:",
      knowledge_df.iloc[n_train+n_val:]["b"].min(),
      knowledge_df.iloc[n_train+n_val:]["b"].max())
