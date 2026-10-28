import pandas as pd

seq = pd.read_csv("data/hdfs/sequences.csv")
labels = pd.read_csv("data/hdfs/preprocessed/anomaly_label.csv")

# label file has "Normal" / "Anomaly", changing to 0 and 1
labels["label"] = (labels["Label"] == "Anomaly").astype(int)
data = seq.merge(labels[["BlockId", "label"]], left_on="block_id", right_on="BlockId")
data = data.drop(columns="BlockId")

print("blocks:", len(data))
print("anomalies:", data["label"].sum(), "(", round(data["label"].mean() * 100, 2), "% )")

# split by time and not random, so model never sees logs from the future
# oldest 70% train, next 10% validation, newest 20% test
# validation is for choosing settings (top k, threshold), so test is used only one time at the end
data = data.sort_values("first_time").reset_index(drop=True)
cut1 = int(len(data) * 0.7)
cut2 = int(len(data) * 0.8)
train = data.iloc[:cut1]
val = data.iloc[cut1:cut2]
test = data.iloc[cut2:]

# models learn only normal behaviour, so keeping only normal blocks for training
train_normal = train[train["label"] == 0]

train.to_csv("data/hdfs/train_all.csv", index=False)
train_normal.to_csv("data/hdfs/train_normal.csv", index=False)
val.to_csv("data/hdfs/val.csv", index=False)
test.to_csv("data/hdfs/test.csv", index=False)

print("train:", len(train), "| normal only:", len(train_normal))
print("val:", len(val), "| anomalies:", val["label"].sum())
print("test:", len(test), "| anomalies:", test["label"].sum())