import sys
import time
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from metrics import evaluate

# run: python src/07_train_transformer.py hdfs  (or bgl)
dataset = sys.argv[1] if len(sys.argv) > 1 else "hdfs"
folder = "data/" + dataset + "/"
name = dataset + "_transformer"

chunk = 20
mask_rate = 0.15
epochs = 10
seeds = [42, 7, 123]
if dataset == "hdfs":
    sample_size = 20000
else:
    sample_size = 5000

train_all = pd.read_csv(folder + "train_normal.csv")
val = pd.read_csv(folder + "val.csv")
test = pd.read_csv(folder + "test.csv")
num_events = pd.read_csv(folder + "templates.csv")["event_id"].max() + 1

# 0 = padding (drain ids start from 1), one extra id for [MASK]
PAD = 0
MASK = num_events
vocab = num_events + 1


# cut long sequences in pieces of 20, short ones get padding
def make_chunks(seq):
    seq = [int(e) for e in seq.split()]
    chunks = []
    for i in range(0, len(seq), chunk):
        c = seq[i:i + chunk]
        chunks.append(c + [PAD] * (chunk - len(c)))
    return chunks


class LogTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, 64, padding_idx=PAD)
        # transformer don't know the order itself, so position embedding is added
        self.pos = nn.Embedding(chunk, 64)
        layer = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(64, vocab)

    def forward(self, x):
        positions = torch.arange(x.shape[1]).unsqueeze(0)
        h = self.embed(x) + self.pos(positions)
        h = self.encoder(h, src_key_padding_mask=(x == PAD))
        return self.out(h)


def train_model(seed):
    torch.manual_seed(seed)
    train = train_all.sample(min(sample_size, len(train_all)), random_state=seed)
    train_chunks = []
    for s in train["events"]:
        train_chunks += make_chunks(s)
    train_chunks = torch.tensor(train_chunks)

    model = LogTransformer(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_chunks))
        total, count = 0, 0
        for i in range(0, len(train_chunks), 256):
            x = train_chunks[order[i:i + 256]]

            # hide 15% of real events like bert, model has to guess from both sides
            hide = (torch.rand(x.shape) < mask_rate) & (x != PAD)
            if hide.sum() == 0:
                continue
            x_in = x.clone()
            x_in[hide] = MASK

            opt.zero_grad()
            logits = model(x_in)
            loss = loss_fn(logits[hide], x[hide])
            loss.backward()
            opt.step()
            total += loss.item() * hide.sum().item()
            count += hide.sum().item()
        print("  epoch", epoch + 1, "loss", round(total / count, 4), "-", round(time.time() - start), "sec")
    return model


# hide each event one by one, count how many the model can't guess in top k
def count_misses(model, df):
    # many chunks repeat, so each unique chunk is scored once
    unique = {}
    seq_ids = []
    for s in df["events"]:
        ids = []
        for c in make_chunks(s):
            key = tuple(c)
            if key not in unique:
                unique[key] = len(unique)
            ids.append(unique[key])
        seq_ids.append(ids)

    model.eval()
    chunks = torch.tensor(list(unique.keys()))
    ranks = torch.zeros(chunks.shape, dtype=torch.long)
    with torch.no_grad():
        for pos in range(chunk):
            for i in range(0, len(chunks), 2048):
                x = chunks[i:i + 2048].clone()
                real = x[:, pos].clone()
                x[:, pos] = MASK
                logits = model(x)[:, pos]
                real_score = logits.gather(1, real.unsqueeze(1))
                ranks[i:i + 2048, pos] = (logits > real_score).sum(dim=1)  # 0 = first guess

    # padding is not real event, never a miss
    ranks[chunks == PAD] = 0
    ranks = ranks.numpy()

    misses = {}
    for k in [5, 9, 15]:
        misses[k] = [sum((ranks[i] >= k).sum() for i in ids) for ids in seq_ids]
    return misses


rows = []
for seed in seeds:
    print("seed", seed)
    model = train_model(seed)
    print("  scoring val and test...")
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
summary = {"model": "transformer", "dataset": dataset, "setting": best}
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