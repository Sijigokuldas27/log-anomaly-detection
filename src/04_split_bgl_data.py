import pandas as pd

data = pd.read_csv("data/bgl/sequences.csv")

# same like hdfs: oldest 70% train, next 10% validation, newest 20% test
data = data.sort_values("first_time").reset_index(drop=True)
cut1 = int(len(data) * 0.7)
cut2 = int(len(data) * 0.8)
train = data.iloc[:cut1]
val = data.iloc[cut1:cut2]
test = data.iloc[cut2:]

# only normal windows for training the models
train_normal = train[train["label"] == 0]

train.to_csv("data/bgl/train_all.csv", index=False)
train_normal.to_csv("data/bgl/train_normal.csv", index=False)
val.to_csv("data/bgl/val.csv", index=False)
test.to_csv("data/bgl/test.csv", index=False)

print("train:", len(train), "| anomalies:", train["label"].sum(),
      "(", round(train["label"].mean() * 100, 2), "% )")
print("normal only:", len(train_normal))
print("val:", len(val), "| anomalies:", val["label"].sum(),
      "(", round(val["label"].mean() * 100, 2), "% )")
print("test:", len(test), "| anomalies:", test["label"].sum(),
      "(", round(test["label"].mean() * 100, 2), "% )")