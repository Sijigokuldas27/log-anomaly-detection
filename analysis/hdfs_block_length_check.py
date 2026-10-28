import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

sys.path.append("src")
from metrics import evaluate

# isolation forest was doing very bad on hdfs (f1 0.014), so checking why
# run from project folder: python analysis/hdfs_block_length_check.py

train = pd.read_csv("data/hdfs/train_normal.csv")
test = pd.read_csv("data/hdfs/test.csv")
all_data = pd.concat([pd.read_csv("data/hdfs/train_all.csv"), test])
num_events = pd.read_csv("data/hdfs/templates.csv")["event_id"].max()


def count_events(sequences):
    X = np.zeros((len(sequences), num_events + 1))
    for i, s in enumerate(sequences):
        for e in s.split():
            X[i, int(e)] += 1
    return X


def length(df):
    return df["events"].str.split().str.len()


# check 1: normal blocks in test are looking different from normal blocks in train?
print("avg events per block")
print("train normal:", round(length(train).mean(), 1))
print("test normal: ", round(length(test[test.label == 0]).mean(), 1))
print("test anomaly:", round(length(test[test.label == 1]).mean(), 1))

# check 2: is the model giving anomalies more unusual score at all?
model = IsolationForest(n_estimators=100, random_state=42)
model.fit(count_events(train["events"]))
scores = -model.score_samples(count_events(test["events"]))
print("\nAUC on time split:", round(roc_auc_score(test["label"], scores), 3), "(0.5 means random guess)")

# check 3: same model with random split, to see if the split is the reason
shuffled = all_data.sample(frac=1, random_state=42)
cut = int(len(shuffled) * 0.8)
r_train = shuffled.iloc[:cut]
r_train = r_train[r_train.label == 0]
r_test = shuffled.iloc[cut:]

model2 = IsolationForest(n_estimators=100, random_state=42)
model2.fit(count_events(r_train["events"]))
train_scores = model2.score_samples(count_events(r_train["events"]))
test_scores = model2.score_samples(count_events(r_test["events"]))
print("AUC on random split:", round(roc_auc_score(r_test["label"], -test_scores), 3))
print()
threshold = np.percentile(train_scores, 1)
evaluate(r_test["label"], (test_scores < threshold).astype(int), "Isolation Forest, random split (1%)")