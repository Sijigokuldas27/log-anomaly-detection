import os
import subprocess
import sys
import time
import pandas as pd
import requests

# one file to see everything
#   python run_all.py        -> shows all my saved results (few seconds, no data needed)
#   python run_all.py full   -> runs the whole project again from the raw logs, then shows results
# full run needs the data in data/hdfs and data/bgl (see README) and takes long on a laptop cpu

steps = [
    ["src/01_parse_hdfs_logs.py"],
    ["src/02_split_hdfs_data.py"],
    ["src/03_parse_bgl_logs.py"],
    ["src/04_split_bgl_data.py"],
    ["src/05_train_isolation_forest.py", "hdfs"],
    ["src/05_train_isolation_forest.py", "bgl"],
    ["src/06_train_lstm.py", "hdfs"],
    ["src/06_train_lstm.py", "bgl"],
    ["src/07_train_transformer.py", "hdfs"],
    ["src/07_train_transformer.py", "bgl"],
    ["src/08_plot_curves.py"],
    ["src/09_speed_test.py"],
    ["src/10_attention_heatmap.py"],
]

needed_files = ["data/hdfs/HDFS.log", "data/hdfs/preprocessed/anomaly_label.csv", "data/bgl/BGL.log"]


def heading(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ollama_running():
    try:
        requests.get("http://localhost:11434", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def run(cmd):
    print("\n>>> python " + " ".join(cmd))
    start = time.time()
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print("\nstopped, this step gave an error:", " ".join(cmd))
        sys.exit(1)
    print("done in", round((time.time() - start) / 60, 1), "min")


def full_run():
    missing = [f for f in needed_files if not os.path.exists(f)]
    if missing:
        print("data is missing:")
        for f in missing:
            print("  ", f)
        print("download HDFS_v1.zip and BGL.zip from https://zenodo.org/records/8196385")
        print("and unzip them into data/hdfs/ and data/bgl/ (see README)")
        sys.exit(1)

    heading("Running the whole project again from the raw logs (this takes long)")
    for cmd in steps:
        run(cmd)

    # llm part needs ollama, if it is not running just skip it
    # 12 and 13 are not run here, 12 makes new answers and 13 is rating by hand,
    # so my rated sheet in results/ is used for the rag comparison
    if ollama_running():
        run(["src/11_llm_explain.py"])
    else:
        print("\nollama is not running, so skipping the llm example (src/11_llm_explain.py)")
    run(["src/14_compare_rag.py"])


def show_results():
    heading("1. How well each model finds anomalies (test set, average of 3 runs)")
    print("F1 and AUC: 1.0 is perfect. The number in brackets is how much the 3 runs differ.\n")
    rows = []
    for dataset in ["hdfs", "bgl"]:
        for model, file in [("Isolation Forest", "isolation_forest_normal"), ("LSTM", "lstm"), ("Transformer", "transformer")]:
            path = "results/" + dataset + "_" + file + "_summary.csv"
            if not os.path.exists(path):
                print("  (not found:", path + ")")
                continue
            s = pd.read_csv(path).iloc[0]
            rows.append({"Dataset": dataset.upper(), "Model": model,
                         "Setting (picked on validation)": s["setting"],
                         "Precision": f"{s['precision_mean']:.3f}",
                         "Recall": f"{s['recall_mean']:.3f}",
                         "F1": f"{s['f1_mean']:.3f} ({s['f1_std']:.3f})",
                         "AUC": f"{s['auc_mean']:.3f}"})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))

    heading("2. Speed of each model on my laptop CPU")
    if os.path.exists("results/speed.csv"):
        speed = pd.read_csv("results/speed.csv")
        table = pd.DataFrame({"Dataset": speed["dataset"].str.upper(),
                              "Model": speed["model"],
                              "Time for 1 sequence (ms)": speed["ms_per_sequence"].round(2),
                              "Sequences per second (batch)": speed["sequences_per_sec"].round(0).astype(int)})
        print(table.to_string(index=False))
    else:
        print("  (not found: results/speed.csv)")

    heading("3. LLM explanations without and with RAG (rated by hand, 30 blocks)")
    print("Each field: 0 wrong, 1 partly, 2 correct. Total is out of 8.\n")
    if os.path.exists("results/llm_rag_summary.csv"):
        names = {"no_rag": "Without RAG", "rag": "With RAG"}
        llm = pd.read_csv("results/llm_rag_summary.csv")
        llm["method"] = llm["method"].map(names)
        llm.columns = ["Answer", "Summary", "Cause", "Severity", "Next step", "Total (of 8)"]
        print(llm.to_string(index=False))

        answers = pd.read_csv("results/llm_rag_comparison.csv")
        split = answers.pivot_table(index="method", columns="real_label", values="total", aggfunc="mean").round(2)
        split = split.rename(index=names, columns={"anomaly": "Real anomaly blocks", "normal": "Normal blocks"})
        split.index.name = "Total (of 8) on"
        split.columns.name = None
        print()
        print(split.to_string())
    else:
        print("  (not found, run: python src/14_compare_rag.py)")

    heading("4. Figures (open these files to see them)")
    figures = [("results/hdfs_curves.png", "ROC and precision-recall curves for HDFS"),
               ("results/bgl_curves.png", "ROC and precision-recall curves for BGL"),
               ("results/hdfs_attention_example.png", "where the Transformer looked in one anomaly block")]
    for f, about in figures:
        print("  " + f + " - " + about + ("" if os.path.exists(f) else "  (not found)"))

    print("\nMore details are in README.md and the notes folder.")


if len(sys.argv) > 1 and sys.argv[1] == "full":
    full_run()
show_results()