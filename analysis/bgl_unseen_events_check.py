import pandas as pd

# lstm and transformer both give ~4000 false alarms on bgl, so checking if
# normal test windows have events which model never saw in training
# run from project folder: python analysis/bgl_unseen_events_check.py

train = pd.read_csv("data/bgl/train_normal.csv")
test = pd.read_csv("data/bgl/test.csv")

# models are trained on 5000 windows only, so checking the sample also
sample = train.sample(5000, random_state=42) if len(train) > 5000 else train


def events_in(df):
    seen = set()
    for s in df["events"]:
        seen.update(s.split())
    return seen


seen_sample = events_in(sample)
seen_all = events_in(train)
print("event types in 5000 sampled windows:", len(seen_sample))
print("event types in all normal training windows:", len(seen_all))

normal_test = test[test["label"] == 0]
for name, seen in [("sample", seen_sample), ("all training", seen_all)]:
    has_new = normal_test["events"].apply(lambda s: any(e not in seen for e in s.split()))
    print("normal test windows with event never seen in " + name + ":",
          has_new.sum(), "of", len(normal_test))