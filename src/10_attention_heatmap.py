import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# shows which events the transformer looked at when it could not guess a hidden event
# uses hdfs model of seed 42 (best on validation)
# run: python src/10_attention_heatmap.py

TOP_K = 5   # same as picked on validation for hdfs
torch.manual_seed(0)

test = pd.read_csv("data/hdfs/test.csv")
templates = pd.read_csv("data/hdfs/templates.csv")
num_events = templates["event_id"].max() + 1
names = dict(zip(templates["event_id"], templates["template"]))
MASK = num_events


# same class as in 07
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


model = LogTransformer(num_events + 1)
model.load_state_dict(torch.load("models/hdfs_transformer.pt"))
model.eval()


def pad20(seq):
    return seq + [0] * (20 - len(seq))


# find first anomaly block (max 20 events) where model can't guess some event
found = None
anomalies = test[test["label"] == 1]
with torch.no_grad():
    for s in anomalies["events"]:
        seq = [int(e) for e in s.split()]
        if len(seq) > 20 or len(seq) < 4:
            continue
        for pos in range(len(seq)):
            x = torch.tensor([pad20(seq)])
            x[0, pos] = MASK
            logits = model(x)[0, pos]
            rank = (logits > logits[seq[pos]]).sum().item()
            if rank >= TOP_K:
                found = (seq, pos, logits.topk(3).indices.tolist())
                break
        if found:
            break

if found is None:
    print("no anomaly with a miss found")
    raise SystemExit

seq, miss_pos, guesses = found
n = len(seq)
print("anomaly block with", n, "events, model missed position", miss_pos + 1)
for i, e in enumerate(seq):
    mark = "  <-- hidden, model could not guess" if i == miss_pos else ""
    print(str(i + 1).rjust(3), "E" + str(e), names.get(e, "?")[:70] + mark)
print("\nmodel top 3 guesses for that position:")
for g in guesses:
    print("   E" + str(g), names.get(g, "?")[:70])

# attention weights from the last layer, for the input with the missed event hidden
x = torch.tensor([pad20(seq)])
x[0, miss_pos] = MASK
pad_mask = (x == 0)
with torch.no_grad():
    h = model.embed(x) + model.pos(torch.arange(20).unsqueeze(0))
    h = model.encoder.layers[0](h, src_key_padding_mask=pad_mask)
    last = model.encoder.layers[1]
    _, att = last.self_attn(h, h, h, key_padding_mask=pad_mask,
                            need_weights=True, average_attn_weights=True)
att = att[0, :n, :n].numpy()

labels = []
for i, e in enumerate(seq):
    if i == miss_pos:
        labels.append(str(i + 1) + " [MASK]")
    else:
        labels.append(str(i + 1) + " E" + str(e))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(att, cmap="Blues", vmin=0)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("looked at (key)")
ax.set_ylabel("event being predicted (query)")
ax.set_title("HDFS anomaly block - transformer attention (last layer)")
# box around the row of the hidden event
ax.add_patch(plt.Rectangle((-0.5, miss_pos - 0.5), n, 1, fill=False, edgecolor="#eb6834", linewidth=2))
fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
plt.tight_layout()
plt.savefig("results/hdfs_attention_example.png", dpi=150)
print("\nsaved results/hdfs_attention_example.png")

# which events the hidden position looked at most
row = att[miss_pos]
top = row.argsort()[::-1][:3]
print("hidden position looked most at:")
for i in top:
    print("   position", i + 1, "E" + str(seq[i]) if i != miss_pos else "[MASK]", "weight", round(float(row[i]), 3))