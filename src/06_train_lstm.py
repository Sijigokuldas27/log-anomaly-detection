import sys
import time
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from metrics import evaluate

# run: python src/06_train_lstm.py hdfs  (or bgl)
dataset = sys.argv[1] if len(sys.argv) > 1 else "hdfs"
folder = "data/" + dataset + "/"
name = dataset + "_lstm"

window = 10
epochs = 5
seeds = [42, 7, 123]
if dataset == "hdfs":
    sample_size = 20000
else:
    sample_size = 5000

train_all = pd.read_csv(folder + "train_normal.csv")
val = pd.read_csv(folder + "val.csv")
test = pd.read_csv(folder + "test.csv")
num_events = pd.read_csv(folder + "templates.csv")["event_id"].max() + 1


# last 10 events -> next event, 0 is padding if less than 10 before
def make_windows(seq):
    seq = [int(e) for e in seq.split()]
    X, y = [], []
    for i in range(1, len(seq)):
        w = seq[max(0, i - window):i]
        X.append([0] * (window - len(w)) + w)
        y.append(seq[i])
    return X, y


class DeepLog(nn.Module):
    def __init__(self, num_events):
        super().__init__()
        self.embed = nn.Embedding(num_events, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, num_events)

    def forward(self, x):
        h, _ = self.lstm(self.embed(x))
        return self.out(h[:, -1])  # last step only


# moved training in function so i can run it for 3 seeds
def train_model(seed):
    torch.manual_seed(seed)
    # small sample is enough, deeplog paper also used small part of hdfs
    train = train_all.sample(min(sample_size, len(train_all)), random_state=seed)
    X_train, y_train = [], []
    for s in train["events"]:
        X, y = make_windows(s)
        X_train += X
        y_train += y
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    # print(X_train.shape)

    model = DeepLog(num_events)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(X_train))
        total = 0
        for i in range(0, len(X_train), 1024):
            idx = order[i:i + 1024]
            opt.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print("  epoch", epoch + 1, "loss", round(total / len(X_train), 4), "-", round(time.time() - start), "sec")
    return model


# for each k, how many next events per sequence are not in top k
def count_misses(model, df):
    # many windows repeat, so model runs once per unique window
    unique = {}
    seq_ids = []
    for s in df["events"]:
        X, y = make_windows(s)
        ids = []
        for w, nxt in zip(X, y):
            key = (tuple(w), nxt)
            if key not in unique:
                unique[key] = len(unique)
            ids.append(unique[key])
        seq_ids.append(ids)

    keys = list(unique.keys())
    windows = torch.tensor([k[0] for k in keys])
    targets = torch.tensor([k[1] for k in keys])

    # rank of real next event (0 = first guess)
    model.eval()
    ranks = []
    with torch.no_grad():
        for i in range(0, len(keys), 4096):
            logits = model(windows[i:i + 4096])
            real = logits.gather(1, targets[i:i + 4096].unsqueeze(1))
            ranks.append((logits > real).sum(dim=1))
    ranks = torch.cat(ranks).numpy()

    misses = {}
    for k in [5, 9, 15]:
        misses[k] = [sum(ranks[i] >= k for i in ids) for ids in seq_ids]
    return misses


rows = []
for seed in seeds:
    print("seed", seed)
    model = train_model(seed)
    val_m = count_misses(model, val)
    test_m = count_misses(model, test)
    if seed == 42:
        torch.save(model.state_dict(), "models/" + name + ".pt")
        first_test_m = test_m

    for k in [5, 9, 15]:
        for split, df, ms in [("val", val, val_m[k]), ("test", test, test_m[k])]:
            auc = roc_auc_score(df["label"], ms)
            # anomaly only if at least m misses
            for m in [1, 2, 5, 10]:
                pred = [int(x >= m) for x in ms]
                r = evaluate(df["label"], pred, "top " + str(k) + ", " + str(m) + "+ misses", show=False)
                r["auc"] = auc
                r["seed"] = seed
                r["split"] = split
                rows.append(r)

results = pd.DataFrame(rows)
results.to_csv("results/" + name + ".csv", index=False)

# pick setting using validation only, not test
val_f1 = results[results["split"] == "val"].groupby("model")["f1"].mean()
best = val_f1.idxmax()
print("\nbest setting on validation:", best, "| val f1:", round(val_f1[best], 3))

# test for that setting, mean and std of 3 seeds
t = results[(results["split"] == "test") & (results["model"] == best)]
summary = {"model": "lstm", "dataset": dataset, "setting": best}
print("test result:")
for col in ["precision", "recall", "f1", "auc"]:
    summary[col + "_mean"] = t[col].mean()
    summary[col + "_std"] = t[col].std()
    print(" ", col, round(t[col].mean(), 3), "+-", round(t[col].std(), 3))
pd.DataFrame([summary]).to_csv("results/" + name + "_summary.csv", index=False)

# test scores of seed 42 for the curves later
best_k = int(best.split(",")[0].replace("top ", ""))
pd.DataFrame({"label": test["label"], "score": first_test_m[best_k]}).to_csv("results/" + name + "_scores.csv", index=False)

print("\nall settings on test (avg of seeds):")
table = results[results["split"] == "test"].groupby("model")[["precision", "recall", "f1", "auc"]].mean()
print(table.round(3).to_string())