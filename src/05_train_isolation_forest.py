import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from metrics import evaluate

# how to run:
#   python src/05_train_isolation_forest.py hdfs
#   python src/05_train_isolation_forest.py bgl
# add "all" in the end to train on all training data (labels not used)
dataset = sys.argv[1] if len(sys.argv) > 1 else "hdfs"
mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
folder = "data/" + dataset + "/"
name = dataset + "_isolation_forest_" + mode

# running 3 times with different seed, to see if result is stable or just luck
SEEDS = [42, 7, 123]

train = pd.read_csv(folder + "train_" + mode + ".csv")
val = pd.read_csv(folder + "val.csv")
test = pd.read_csv(folder + "test.csv")
num_events = pd.read_csv(folder + "templates.csv")["event_id"].max()


# each sequence becomes a vector: how many times each event is coming
def count_events(sequences):
    X = np.zeros((len(sequences), num_events + 1))
    for i, s in enumerate(sequences):
        for e in s.split():
            X[i, int(e)] += 1
    return X


def make_features(X, kind):
    if kind == "counts":
        return X
    if kind == "binary":
        # only if event happened or not, not how many times
        return (X > 0).astype(float)
    if kind == "ratio":
        # share of each event in the sequence, so length is not affecting much
        return X / X.sum(axis=1, keepdims=True).clip(min=1)


X_train_raw = count_events(train["events"])
X_val_raw = count_events(val["events"])
X_test_raw = count_events(test["events"])

rows = []
first_seed_scores = {}

for seed in SEEDS:
    for kind in ["counts", "binary", "ratio"]:
        X_train = make_features(X_train_raw, kind)
        model = IsolationForest(n_estimators=100, random_state=seed)
        model.fit(X_train)

        # sklearn gives lower score for unusual, so using minus to get anomaly score
        train_scores = -model.score_samples(X_train)
        val_scores = -model.score_samples(make_features(X_val_raw, kind))
        test_scores = -model.score_samples(make_features(X_test_raw, kind))
        if seed == SEEDS[0]:
            first_seed_scores[kind] = test_scores

        # threshold is taken only from train scores, labels are not used for it
        for pct in [1, 5, 10]:
            threshold = np.percentile(train_scores, 100 - pct)
            setting = kind + ", " + str(pct) + "%"
            for split, df, sc in [("val", val, val_scores), ("test", test, test_scores)]:
                r = evaluate(df["label"], (sc > threshold).astype(int), setting, show=False)
                r["auc"] = roc_auc_score(df["label"], sc)
                r["seed"] = seed
                r["split"] = split
                rows.append(r)
    print("seed", seed, "done")

results = pd.DataFrame(rows)
results.to_csv("results/" + name + ".csv", index=False)

# choosing setting with best average f1 on validation, test is not used for choosing
val_f1 = results[results["split"] == "val"].groupby("model")["f1"].mean()
best = val_f1.idxmax()
print("\nbest setting on validation:", best, "| val f1:", round(val_f1[best], 3))

# test result only for that setting, mean and std over the 3 seeds
t = results[(results["split"] == "test") & (results["model"] == best)]
summary = {"model": "isolation_forest", "dataset": dataset, "mode": mode, "setting": best}
print("test result:")
for col in ["precision", "recall", "f1", "auc"]:
    summary[col + "_mean"] = t[col].mean()
    summary[col + "_std"] = t[col].std()
    print(" ", col, round(t[col].mean(), 3), "+-", round(t[col].std(), 3))
pd.DataFrame([summary]).to_csv("results/" + name + "_summary.csv", index=False)

# saving test scores of first seed, will use later for pr and roc curves
kind = best.split(",")[0]
scores = pd.DataFrame({"label": test["label"], "score": first_seed_scores[kind]})
scores.to_csv("results/" + name + "_scores.csv", index=False)

print("\nall settings on test (mean of seeds), only for the ablation table:")
table = results[results["split"] == "test"].groupby("model")[["precision", "recall", "f1", "auc"]].mean()
print(table.round(3).to_string())