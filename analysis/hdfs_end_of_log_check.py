import sys
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.append("src")
from metrics import evaluate

# test blocks are shorter than train blocks. i thought they are blocks not finished when the log ended
# checking that here, and also where the false alarms of each model come from
# run from project folder: python analysis/hdfs_end_of_log_check.py

train = pd.read_csv("data/hdfs/train_normal.csv")
test = pd.read_csv("data/hdfs/test.csv")
templates = pd.read_csv("data/hdfs/templates.csv")
num_events = templates["event_id"].max()
names = dict(zip(templates["event_id"], templates["template"]))

# block is stored when "addStoredBlock: blockMap updated" comes. no such line = not finished
stored_ids = []
for e, t in names.items():
    if "addStoredBlock: blockMap updated" in t:
        stored_ids.append(e)
print("stored event ids:", stored_ids)

lengths = []
unfinished = []
for s in test["events"]:
    ev = [int(e) for e in s.split()]
    lengths.append(len(ev))
    stored = 0
    for e in ev:
        if e in stored_ids:
            stored += 1
    unfinished.append(1 if stored == 0 else 0)
test["length"] = lengths
test["unfinished"] = unfinished

# cut test period in 10 parts by time, part 10 = end of the log
test = test.sort_values("first_time").reset_index(drop=True)
test["part"] = test.index * 10 // len(test) + 1

print("\ntest period in 10 parts by time")
print("part | blocks | avg events | unfinished % | anomaly %")
for p in range(1, 11):
    d = test[test["part"] == p]
    print(p, "|", len(d), "|", round(d["length"].mean(), 1), "|",
          round(d["unfinished"].mean() * 100, 1), "|", round(d["label"].mean() * 100, 1))

normal = test[test["label"] == 0]
print("\nunfinished normal blocks:", normal["unfinished"].sum(), "of", len(normal))
print("of them in the last part:", len(normal[(normal["unfinished"] == 1) & (normal["part"] == 10)]))

# flags of each model, same settings picked on validation
# score files are in the same order as test.csv, so i read test.csv again and match by block id
order = pd.read_csv("data/hdfs/test.csv")[["block_id"]]
lstm = pd.read_csv("results/hdfs_lstm_scores.csv")
trans = pd.read_csv("results/hdfs_transformer_scores.csv")
order["LSTM"] = (lstm["score"] >= 1).astype(int)          # top 5, 1+ misses
order["Transformer"] = (trans["score"] >= 1).astype(int)  # top 5, 1+ misses


# isolation forest again: ratio features, 10%, seed 42, same as in 05
def ratio_features(sequences):
    X = np.zeros((len(sequences), num_events + 1))
    for i, s in enumerate(sequences):
        ev = s.split()
        for e in ev:
            X[i, int(e)] += 1
        X[i] = X[i] / len(ev)
    return X


model = IsolationForest(n_estimators=100, random_state=42)
model.fit(ratio_features(train["events"]))
train_scores = -model.score_samples(ratio_features(train["events"]))
threshold = np.percentile(train_scores, 90)
test_scores = -model.score_samples(ratio_features(test["events"]))
test["Isolation Forest"] = (test_scores > threshold).astype(int)

test = test.merge(order, on="block_id")
models = ["LSTM", "Transformer", "Isolation Forest"]

print("\nfalse alarms (normal blocks flagged)")
print("model | false alarms | unfinished | in last part")
for m in models:
    fa = test[(test[m] == 1) & (test["label"] == 0)]
    print(m, "|", len(fa), "|", fa["unfinished"].sum(), "|", len(fa[fa["part"] == 10]))

# a real system checks a block after it is finished, so what if we leave out
# the unfinished blocks from the last part (normal and anomaly both)
keep = ~((test["unfinished"] == 1) & (test["part"] == 10))
print("\nleaving out", (~keep).sum(), "unfinished blocks from the last part:")
for m in models:
    evaluate(test["label"], test[m], m + ", all test blocks")
    evaluate(test["label"][keep], test[m][keep], m + ", without them")
    print()

# so where do the isolation forest false alarms come from?
fa = test[(test["Isolation Forest"] == 1) & (test["label"] == 0)]
print("isolation forest false alarms, a closer look")
print("avg events: false alarms", round(fa["length"].mean(), 1),
      "| all normal test", round(normal["length"].mean(), 1),
      "| normal train", round(train["events"].str.split().str.len().mean(), 1))
print("false alarms in each part:", fa["part"].value_counts().sort_index().to_dict())


# forest only sees which events and how many times, not the order
# so i write each block as "event: how many times" and count same ones
def pattern(s):
    counts = Counter(int(e) for e in s.split())
    return tuple(sorted(counts.items()))


train_patterns = Counter()
for s in train["events"]:
    train_patterns[pattern(s)] += 1
fa_patterns = Counter()
for s in fa["events"]:
    fa_patterns[pattern(s)] += 1

never_seen = 0
for p, c in fa_patterns.items():
    if p not in train_patterns:
        never_seen += c
print("false alarms with a pattern never seen in train:", never_seen, "of", len(fa))

print("\n3 most common false alarm patterns:")
for p, c in fa_patterns.most_common(3):
    print("\n", c, "false alarms | same pattern in normal train:", train_patterns[p])
    for e, k in p:
        print("   ", k, "x", names[e][:70])