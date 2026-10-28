import random
import time
import importlib.util
import pandas as pd

# makes the sheet for rating llm explanations by hand
# 24 real anomalies + 6 false alarms, all flagged by the transformer (seed 42)
# each block is explained 2 times: without rag and with rag
# A and B are mixed randomly so i don't know which one is rag when rating (blind)
# run: python src/12_llm_rating_sheet.py   (takes around 20-30 min)

n_anomaly = 24
n_false_alarm = 6
max_events = 60   # very long blocks are slow for the llm
random.seed(42)

# using the functions from 11_llm_explain.py (file name starts with number so normal import not working)
spec = importlib.util.spec_from_file_location("llm", "src/11_llm_explain.py")
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)

test = pd.read_csv("data/hdfs/test.csv")
scores = pd.read_csv("results/hdfs_transformer_scores.csv")
test["flagged"] = (scores["score"] >= 1).astype(int)   # 1+ miss, same as picked on validation


# one block per different set of events, so we get many kinds of problems, not same one 30 times
def pick(df, n):
    rows = list(df.index)
    random.shuffle(rows)
    seen = set()
    picked = []
    for i in rows:
        seq = [int(e) for e in df.loc[i, "events"].split()]
        kind = frozenset(seq)
        if len(seq) > max_events or kind in seen:
            continue
        seen.add(kind)
        picked.append(i)
        if len(picked) == n:
            break
    return picked


true_alarms = test[(test["flagged"] == 1) & (test["label"] == 1)]
false_alarms = test[(test["flagged"] == 1) & (test["label"] == 0)]
chosen = pick(true_alarms, n_anomaly) + pick(false_alarms, n_false_alarm)
random.shuffle(chosen)
print("blocks picked:", len(chosen))


def as_text(ans):
    if "raw_answer" in ans:
        return {"summary": ans["raw_answer"], "cause": "", "severity": "", "next": ""}
    return {"summary": str(ans.get("summary", "")), "cause": str(ans.get("likely_cause", "")),
            "severity": str(ans.get("severity", "")), "next": str(ans.get("next_step", ""))}


sheet = []
key = []
times = []
for n, i in enumerate(chosen):
    seq = [int(e) for e in test.loc[i, "events"].split()]

    start = time.time()
    no_rag = as_text(llm.explain(seq))
    times.append(time.time() - start)

    found = llm.find_notes(seq)
    start = time.time()
    with_rag = as_text(llm.explain(seq, "\n\n".join(found)))
    times.append(time.time() - start)

    # random which one is A and which is B
    if random.random() < 0.5:
        a, b, a_is = no_rag, with_rag, "no_rag"
    else:
        a, b, a_is = with_rag, no_rag, "rag"

    row = {"id": n + 1, "block_id": test.loc[i, "block_id"],
           "real_label": "anomaly" if test.loc[i, "label"] == 1 else "normal",
           "log_events": llm.block_text(seq)}
    for name, ans in [("A", a), ("B", b)]:
        row[name + "_summary"] = ans["summary"]
        row[name + "_cause"] = ans["cause"]
        row[name + "_severity"] = ans["severity"]
        row[name + "_next_step"] = ans["next"]
    # empty columns to fill by hand: 0 wrong, 1 partly, 2 correct
    for name in ["A", "B"]:
        for f in ["summary", "cause", "severity", "next_step"]:
            row[name + "_" + f + "_score"] = ""
        row[name + "_overall"] = ""   # correct / partly / wrong
    row["comment"] = ""
    sheet.append(row)
    key.append({"id": n + 1, "A_is": a_is, "notes_found": " | ".join(x.split("\n")[0] for x in found)})

    # saving every time, so if it stops in middle nothing is lost
    pd.DataFrame(sheet).to_csv("results/llm_rating_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv("results/llm_rating_key.csv", index=False)
    print(n + 1, "of", len(chosen), "done,", round(sum(times[-2:]), 1), "sec")

print("\naverage time per answer:", round(sum(times) / len(times), 1), "sec")
print("saved results/llm_rating_sheet.csv (to fill) and results/llm_rating_key.csv (dont open before rating)")