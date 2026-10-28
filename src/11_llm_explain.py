import json
import time
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# llm runs local with ollama, logs dont go outside
# using openai style api, so for other llm only url and model name need change
url = "http://localhost:11434/v1/chat/completions"
model_name = "llama3.2:3b"

templates = pd.read_csv("data/hdfs/templates.csv")
names = dict(zip(templates["event_id"], templates["template"]))

system_prompt = """You are helping a support engineer. Below are the log events of one HDFS block, in order.
Our model said this block is an anomaly.
Give the answer only as JSON with these keys:
summary - what happened to the block, in one sentence
likely_cause - the most possible reason
severity - low, medium or high
next_step - what the engineer should check first
If the reason is not clear from the logs, write that. Don't add anything which is not in the logs."""


def ask(messages):
    body = {"model": model_name, "messages": messages, "temperature": 0}
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# runbook for rag, each note starts with "# title"
runbook = open("rag/hdfs_runbook.txt", encoding="utf-8").read()
notes = ["# " + n.strip() for n in runbook.split("# ")[1:]]
vectorizer = TfidfVectorizer(stop_words="english")
notes_matrix = vectorizer.fit_transform(notes)


# find the runbook notes most similar to the log messages of the block (tf-idf search)
def find_notes(events, k=3):
    query = " ".join(names.get(e, "") for e in set(events))
    sims = cosine_similarity(vectorizer.transform([query]), notes_matrix)[0]
    best = sims.argsort()[::-1][:k]
    return [notes[i] for i in best]


# change event ids to the log text, numbered, this is what llm reads
def block_text(events):
    lines = []
    for i, e in enumerate(events):
        lines.append(str(i + 1) + ". " + names.get(e, "unknown event"))
    return "\n".join(lines)


def explain(events, context=None):
    msg = "Log events:\n" + block_text(events)
    if context:
        # for rag, runbook notes go before the logs
        msg = "Runbook notes:\n" + context + "\n\n" + msg
    answer = ask([{"role": "system", "content": system_prompt},
                  {"role": "user", "content": msg}])
    # sometimes the model writes extra text with the json, so only taking {...}
    try:
        answer_json = answer[answer.index("{"):answer.rindex("}") + 1]
        return json.loads(answer_json)
    except ValueError:
        return {"raw_answer": answer}


if __name__ == "__main__":
    # first test with same block as attention example (the delete error one)
    test = pd.read_csv("data/hdfs/test.csv")
    error_ids = [e for e, t in names.items() if "Unexpected error trying to delete block" in t]

    block = None
    for s, label in zip(test["events"], test["label"]):
        seq = [int(e) for e in s.split()]
        if label == 1 and len(seq) <= 20 and any(e in error_ids for e in seq):
            block = seq
            break
    if block is None:
        block = [int(e) for e in test[test["label"] == 1]["events"].iloc[0].split()]

    print(block_text(block))

    print("\n--- without rag ---")
    start = time.time()
    print(json.dumps(explain(block), indent=2))
    print("time:", round(time.time() - start, 1), "sec")

    found = find_notes(block)
    print("\n--- with rag ---")
    print("notes found:", [n.split("\n")[0] for n in found])
    start = time.time()
    print(json.dumps(explain(block, "\n\n".join(found)), indent=2))
    print("time:", round(time.time() - start, 1), "sec")