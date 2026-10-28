import re
import time
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

# drain settings
config = TemplateMinerConfig()
config.profiling_enabled = False
config.drain_sim_th = 0.5
config.drain_depth = 4

# masking blk id, ip and numbers, else drain gives new template almost for every line
config.masking_instructions = [
    MaskingInstruction(r"blk_-?\d+", "BLK"),
    MaskingInstruction(r"/?\d+\.\d+\.\d+\.\d+(:\d+)?", "IP"),
    MaskingInstruction(r"\b\d+\b", "NUM"),
]
miner = TemplateMiner(config=config)

blk_re = re.compile(r"blk_-?\d+")
events = {}      # blk id -> list of event ids
first_seen = {}  # blk id -> time of first line

start = time.time()
n = 0

with open("data/hdfs/HDFS.log", encoding="utf-8", errors="ignore") as f:
    for line in f:
        n += 1
        parts = line.strip().split(" ", 5)
        # print(parts)
        if len(parts) < 6:
            continue

        msg = parts[5]
        eid = miner.add_log_message(msg)["cluster_id"]

        # some lines have more than 1 block (like delete request for many blocks), so add to all of them
        for b in set(blk_re.findall(msg)):
            if b not in events:
                events[b] = []
                first_seen[b] = int(parts[0] + parts[1])
            events[b].append(eid)

        if n % 1000000 == 0:
            print(n, "lines done", round(time.time() - start), "sec")

print("total lines", n)
print("blocks", len(events))
print("templates", len(miner.drain.clusters))

# save templates
temp = []
for c in miner.drain.clusters:
    temp.append([c.cluster_id, c.size, c.get_template()])
templates = pd.DataFrame(temp, columns=["event_id", "count", "template"])
templates = templates.sort_values("count", ascending=False)
templates.to_csv("data/hdfs/templates.csv", index=False)
print(templates.head(10))

# save sequence of each block
temp = []
for b in events:
    temp.append([b, first_seen[b], " ".join(str(x) for x in events[b])])
df = pd.DataFrame(temp, columns=["block_id", "first_time", "events"])
df.to_csv("data/hdfs/sequences.csv", index=False)
print("saved")