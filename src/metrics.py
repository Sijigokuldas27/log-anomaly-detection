from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# same function used for all 3 models, so all are measured in same way
# show=False when running many settings, otherwise too much printing
def evaluate(y_true, y_pred, name, show=True):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if show:
        print(name)
        print("precision:", round(p, 3), "| recall:", round(r, 3), "| f1:", round(f1, 3))
        print("caught", tp, "of", tp + fn, "anomalies, with", fp, "false alarms")
    return {"model": name, "precision": p, "recall": r, "f1": f1}