import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

# how fast each model can check a log sequence
# latency = one sequence at a time (like real time, a new block comes and we check it)
# throughput = many sequences together in batch
# run: python src/09_speed_test.py

N_LATENCY = 300     # sequences checked one by one
N_BATCH = 3000      # sequences checked together
torch.set_num_threads(4)


# same model classes as in 06 and 07
class DeepLog(nn.Module):
    def __init__(self, num_events):
        super().__init__()
        self.embed = nn.Embedding(num_events, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, num_events)

    def forward(self, x):
        h, _ = self.lstm(self.embed(x))
        return self.out(h[:, -1])


class LogTransformer(nn.Module):
    def __init__(self, vocab, chunk=20):
        super().__init__()
        self.embed = nn.Embedding(vocab, 64, padding_idx=0)
        self.pos = nn.Embedding(chunk, 64)
        layer = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=128,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(64, vocab)

    def forward(self, x):
        positions = torch.arange(x.shape[1]).unsqueeze(0)
        h = self.embed(x) + self.pos(positions)
        h = self.encoder(h, src_key_padding_mask=(x == 0))
        return self.out(h)


def lstm_inputs(seqs):
    X = []
    for seq in seqs:
        seq = [int(e) for e in seq.split()]
        for i in range(1, len(seq)):
            w = seq[max(0, i - 10):i]
            X.append([0] * (10 - len(w)) + w)
    return torch.tensor(X)


# every chunk of 20 is copied 20 times, each copy has one different position hidden
def transformer_inputs(seqs, mask_id):
    X = []
    for seq in seqs:
        seq = [int(e) for e in seq.split()]
        for i in range(0, len(seq), 20):
            c = seq[i:i + 20]
            c = c + [0] * (20 - len(c))
            for pos in range(20):
                if c[pos] == 0:
                    continue
                x = list(c)
                x[pos] = mask_id
                X.append(x)
    return torch.tensor(X)


def run(model_fn, seqs_one, seqs_batch):
    # first call is always slow (warm up), so not counting it
    model_fn(seqs_one[:5])
    # one by one
    start = time.time()
    for s in seqs_one:
        model_fn([s])
    latency_ms = (time.time() - start) / len(seqs_one) * 1000
    # all together, in parts of 500
    start = time.time()
    for i in range(0, len(seqs_batch), 500):
        model_fn(seqs_batch[i:i + 500])
    per_sec = len(seqs_batch) / (time.time() - start)
    return latency_ms, per_sec


rows = []
for dataset in ["hdfs", "bgl"]:
    folder = "data/" + dataset + "/"
    train = pd.read_csv(folder + "train_normal.csv")
    test = pd.read_csv(folder + "test.csv")
    num_events = pd.read_csv(folder + "templates.csv")["event_id"].max() + 1

    seqs = list(test["events"].sample(min(N_BATCH, len(test)), random_state=1))
    seqs_one = seqs[:N_LATENCY]
    avg_len = np.mean([len(s.split()) for s in seqs])
    print(dataset, "- avg events per sequence:", round(avg_len, 1))

    # isolation forest, features picked on validation (ratio for hdfs, binary for bgl)
    def features(s_list):
        X = np.zeros((len(s_list), num_events))
        for i, s in enumerate(s_list):
            for e in s.split():
                X[i, int(e)] += 1
        if dataset == "hdfs":
            return X / X.sum(axis=1, keepdims=True).clip(min=1)
        return (X > 0).astype(float)

    forest = IsolationForest(n_estimators=100, random_state=42)
    forest.fit(features(list(train["events"].sample(min(50000, len(train)), random_state=1))))
    lat, ps = run(lambda s: forest.score_samples(features(s)), seqs_one, seqs)
    rows.append([dataset, "Isolation Forest", lat, ps])
    print("  isolation forest done")

    lstm = DeepLog(num_events)
    lstm.load_state_dict(torch.load("models/" + dataset + "_lstm.pt"))
    lstm.eval()

    def lstm_fn(s):
        with torch.no_grad():
            X = lstm_inputs(s)
            if len(X) > 0:
                lstm(X)

    lat, ps = run(lstm_fn, seqs_one, seqs)
    rows.append([dataset, "LSTM", lat, ps])
    print("  lstm done")

    tr = LogTransformer(num_events + 1)
    tr.load_state_dict(torch.load("models/" + dataset + "_transformer.pt"))
    tr.eval()

    def tr_fn(s):
        with torch.no_grad():
            X = transformer_inputs(s, num_events)
            for i in range(0, len(X), 4096):
                tr(X[i:i + 4096])

    lat, ps = run(tr_fn, seqs_one, seqs)
    rows.append([dataset, "Transformer", lat, ps])
    print("  transformer done")

speed = pd.DataFrame(rows, columns=["dataset", "model", "ms_per_sequence", "sequences_per_sec"])
speed.to_csv("results/speed.csv", index=False)
print()
print(speed.round(2).to_string(index=False))