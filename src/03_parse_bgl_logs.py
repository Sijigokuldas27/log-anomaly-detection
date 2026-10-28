import time
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

WINDOW = 100  # lines per window

# drain settings, same as hdfs
config = TemplateMinerConfig()
config.profiling_enabled = False
config.drain_sim_th = 0.5
config.drain_depth = 4

# bgl has hex addresses also, so masking them with ip and numbers
config.masking_instructions = [
    MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
    MaskingInstruction(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "IP"),
    MaskingInstruction(r"\b\d+\b", "NUM"),
]
miner = TemplateMiner(config=config)

events = []
labels = []
times = []

start = time.time()
with open("data/bgl/BGL.log", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # format: label, unix time, date, node, time, node, type, component, level, message
        parts = line.strip().split(" ", 9)
        if len(parts) < 10:
            continue

        # first column is "-" when line is normal, otherwise it is the alert name
        labels.append(0 if parts[0] == "-" else 1)
        times.append(int(parts[1]))
        events.append(miner.add_log_message(parts[9])["cluster_id"])

        if len(events) % 1000000 == 0:
            print(len(events), "lines done", round(time.time() - start), "sec")

print("total lines", len(events))
print("alert lines", sum(labels))
print("templates", len(miner.drain.clusters))

# save templates
temp = []
for c in miner.drain.clusters:
    temp.append([c.cluster_id, c.size, c.get_template()])
templates = pd.DataFrame(temp, columns=["event_id", "count", "template"])
templates = templates.sort_values("count", ascending=False)
templates.to_csv("data/bgl/templates.csv", index=False)
print(templates.head(10))

# bgl don't have block id like hdfs, so cutting the log in windows of 100 lines
# window is anomaly if even one line inside it is alert
temp = []
for i in range(0, len(events), WINDOW):
    w = events[i:i + WINDOW]
    w_label = max(labels[i:i + WINDOW])
    temp.append([i // WINDOW, times[i], " ".join(str(x) for x in w), w_label])
df = pd.DataFrame(temp, columns=["window_id", "first_time", "events", "label"])
df.to_csv("data/bgl/sequences.csv", index=False)

print("windows", len(df))
print("anomaly windows", df["label"].sum(), "(", round(df["label"].mean() * 100, 2), "% )")