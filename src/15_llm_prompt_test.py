import random
import importlib.util
import pandas as pd

# small test of a better prompt on 12 of the blocks i already rated
# (all 6 normal ones + 6 real anomalies)
# old prompt said "our model said this block is an anomaly", so the llm never said a block looks fine
# new prompt says the detector can be wrong, and asks the llm first if the block looks normal
# no hand rating here: looks_normal yes/no can be checked with the real label
# run: python src/15_llm_prompt_test.py   (needs ollama, took around 45 min on my laptop)

random.seed(42)

spec = importlib.util.spec_from_file_location("llm", "src/11_llm_explain.py")
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)

# explain() in 11 uses llm.system_prompt, so changing it here is enough
llm.system_prompt = """You are helping a support engineer. Below are the log events of one HDFS block, in order.
An anomaly detector flagged this block, but the detector is sometimes wrong.
First decide if the block really looks like a problem. Many blocks are normal: the block is written,
stored on the machines, maybe copied to another machine or deleted later, and nothing failed.
Give the answer only as JSON with these keys:
looks_normal - yes or no
summary - what happened to the block, in one sentence
likely_cause - the most possible reason, or none if it looks normal
severity - none, low, medium or high
next_step - what the engineer should check first, or nothing if it looks normal
Don't add anything which is not in the logs."""

sheet = pd.read_csv("results/llm_rating_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
test = pd.read_csv("data/hdfs/test.csv")
events_of = dict(zip(test["block_id"], test["events"]))

normal_ids = list(sheet[sheet["real_label"] == "normal"]["id"])
anomaly_ids = list(sheet[sheet["real_label"] == "anomaly"]["id"])
chosen = normal_ids + random.sample(anomaly_ids, 6)
print("blocks:", len(chosen), "(" + str(len(normal_ids)) + " normal, 6 anomaly)")

rows = []
for n, i in enumerate(chosen):
    r = sheet[sheet["id"] == i].iloc[0]
    seq = [int(e) for e in events_of[r["block_id"]].split()]
    found = llm.find_notes(seq)

    for method in ["no_rag", "rag"]:
        context = "\n\n".join(found) if method == "rag" else None
        ans = llm.explain(seq, context)
        said = str(ans.get("looks_normal", "")).strip().lower()
        # sometimes the model gives true/false instead of yes/no
        if said == "true":
            said = "yes"
        if said == "false":
            said = "no"
        rows.append({"id": i, "real_label": r["real_label"], "method": method,
                     "looks_normal": said,
                     "summary": ans.get("summary", ans.get("raw_answer", "")),
                     "severity": str(ans.get("severity", "")).lower()})
        print(n + 1, "of", len(chosen), "| block", i, "|", r["real_label"], "|", method,
              "| looks normal:", said, "|", rows[-1]["summary"])

    # save after every block, so nothing is lost if it stops
    pd.DataFrame(rows).to_csv("results/llm_prompt_test.csv", index=False, encoding="utf-8-sig")

res = pd.DataFrame(rows)
# right answer: normal block -> looks_normal yes, anomaly block -> no
res["right"] = ((res["real_label"] == "normal") & (res["looks_normal"] == "yes")) | \
               ((res["real_label"] == "anomaly") & (res["looks_normal"] == "no"))
res.to_csv("results/llm_prompt_test.csv", index=False, encoding="utf-8-sig")

print("\nnew prompt: how many times the llm decided right (normal or problem)")
table = res.pivot_table(index="method", columns="real_label", values="right", aggfunc="sum")
table.columns = [c + " blocks (of 6)" for c in table.columns]
print(table.to_string())

print("\nseverity said by the new prompt")
print(pd.crosstab(res["method"], res["severity"]).to_string())

print("\nsaved results/llm_prompt_test.csv")