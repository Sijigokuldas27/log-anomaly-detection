import pandas as pd

# helps me rate the llm answers one by one in the terminal, easier than excel
# shows the logs and answer A and B, then i type the scores
# scores: 0 wrong, 1 partly, 2 correct (for summary, cause, severity, next step)
# can stop any time with "q", next time it starts from the first row not rated
# run: python src/13_rate_answers.py

path = "results/llm_rating_sheet.csv"
sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
fields = ["summary", "cause", "severity", "next_step"]


def overall(total):
    if total >= 7:
        return "correct"
    if total >= 3:
        return "partly"
    return "wrong"


def ask_scores(name):
    while True:
        s = input("  scores for " + name + " (4 numbers 0/1/2 for summary cause severity next_step, like 2 1 1 2) or q: ").strip()
        if s.lower() == "q":
            return None
        parts = s.split()
        if len(parts) == 4 and all(p in ["0", "1", "2"] for p in parts):
            return [int(p) for p in parts]
        print("  please type 4 numbers, each 0, 1 or 2")


for i in range(len(sheet)):
    if sheet.loc[i, "A_overall"] != "" and sheet.loc[i, "B_overall"] != "":
        continue   # already rated

    row = sheet.loc[i]
    print("\n" + "=" * 70)
    print("block", row["id"], "of", len(sheet), "| real label:", row["real_label"])
    print("=" * 70)
    print(row["log_events"])
    for name in ["A", "B"]:
        print("\n--- answer " + name + " ---")
        print("summary  :", row[name + "_summary"])
        print("cause    :", row[name + "_cause"])
        print("severity :", row[name + "_severity"])
        print("next step:", row[name + "_next_step"])
    print()

    stop = False
    for name in ["A", "B"]:
        scores = ask_scores(name)
        if scores is None:
            stop = True
            break
        for f, s in zip(fields, scores):
            sheet.loc[i, name + "_" + f + "_score"] = str(s)
        sheet.loc[i, name + "_overall"] = overall(sum(scores))
        print("  " + name + " total", sum(scores), "of 8 ->", overall(sum(scores)))
    if stop:
        break
    sheet.loc[i, "comment"] = input("  comment (or just press enter): ").strip()

    # save after every block
    sheet.to_csv(path, index=False, encoding="utf-8-sig")

done = ((sheet["A_overall"] != "") & (sheet["B_overall"] != "")).sum()
sheet.to_csv(path, index=False, encoding="utf-8-sig")
print("\nrated", done, "of", len(sheet), "blocks, saved")