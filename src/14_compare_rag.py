import pandas as pd

# after rating all blocks, this opens the key and puts the scores back to no_rag and rag
# (in the sheet they are only A and B, so the rating was blind)
# run: python src/14_compare_rag.py

sheet = pd.read_csv("results/llm_rating_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
key = pd.read_csv("results/llm_rating_key.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
sheet = sheet.merge(key, on="id")
fields = ["summary", "cause", "severity", "next_step"]

# make one row per answer, so each block gives 2 rows (no_rag and rag)
rows = []
for _, r in sheet.iterrows():
    for side in ["A", "B"]:
        if r["A_is"] == "rag":
            method = "rag" if side == "A" else "no_rag"
        else:
            method = "no_rag" if side == "A" else "rag"
        row = {"id": int(r["id"]), "real_label": r["real_label"], "method": method,
               "said_severity": r[side + "_severity"].strip().lower(),
               "overall": r[side + "_overall"]}
        for f in fields:
            row[f] = int(r[side + "_" + f + "_score"])
        rows.append(row)
answers = pd.DataFrame(rows)
answers["total"] = answers[fields].sum(axis=1)

print("blocks rated:", len(sheet), "| anomaly:", (sheet["real_label"] == "anomaly").sum(),
      "| normal:", (sheet["real_label"] == "normal").sum())

print("\naverage score per field (0 to 2), total is out of 8")
table = answers.groupby("method")[fields + ["total"]].mean().round(2)
print(table.to_string())

print("\noverall labels")
print(pd.crosstab(answers["method"], answers["overall"]).to_string())

print("\naverage total, split by real label")
print(answers.pivot_table(index="method", columns="real_label", values="total", aggfunc="mean").round(2).to_string())

print("\nwhat severity the llm said")
print(pd.crosstab(answers["method"], answers["said_severity"]).to_string())

# which one was better on each block
wide = answers.pivot(index="id", columns="method", values="total")
print("\nper block: rag better", (wide["rag"] > wide["no_rag"]).sum(),
      "| no_rag better", (wide["no_rag"] > wide["rag"]).sum(),
      "| same", (wide["rag"] == wide["no_rag"]).sum())

# on normal blocks the right answer is "nothing wrong", check if any answer got even 3 of 8
normal = answers[answers["real_label"] == "normal"]
print("normal blocks where answer got 3 or more:", (normal["total"] >= 3).sum(), "of", len(normal))

answers.to_csv("results/llm_rag_comparison.csv", index=False)
table.to_csv("results/llm_rag_summary.csv")
print("\nsaved results/llm_rag_comparison.csv and results/llm_rag_summary.csv")