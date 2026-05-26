#!/usr/bin/env python3
"""
evaluate.py - Simple evaluation script for agent outputs.

Compares `support_tickets/output.csv` against a gold CSV (optional) and
computes classification metrics and simple calibration (Brier, ECE) for the
`status` field and action-level precision/recall when `gold_actions` are
available.

Usage:
    python evaluate.py [--gold path/to/gold.csv]

If no gold is provided, the script will try `support_tickets/sample_support_tickets.csv`.
"""

import csv
import json
import sys
from statistics import mean
from pathlib import Path


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_parse_actions(raw):
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def classification_metrics(gold_labels, pred_labels):
    classes = sorted(set(gold_labels) | set(pred_labels))
    results = {}
    for cls in classes:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for g in gold_labels if g == cls)}
    # Macro averages
    macro = {"precision": mean([v["precision"] for v in results.values()]),
             "recall": mean([v["recall"] for v in results.values()]),
             "f1": mean([v["f1"] for v in results.values()])}
    return results, macro


def brier_score(pred_probs, labels):
    # labels are 0/1, pred_probs in [0,1]
    return mean([(p - l) ** 2 for p, l in zip(pred_probs, labels)]) if pred_probs else float("nan")


def expected_calibration_error(pred_probs, labels, n_bins=10):
    # ECE: weighted average |acc - conf| across bins
    if not pred_probs:
        return float("nan")
    bins = [0 for _ in range(n_bins)]
    bin_acc = [0.0 for _ in range(n_bins)]
    bin_conf = [0.0 for _ in range(n_bins)]
    for p, l in zip(pred_probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b] += 1
        bin_acc[b] += l
        bin_conf[b] += p
    total = len(pred_probs)
    ece = 0.0
    for i in range(n_bins):
        if bins[i] == 0:
            continue
        acc = bin_acc[i] / bins[i]
        conf = bin_conf[i] / bins[i]
        ece += (bins[i] / total) * abs(acc - conf)
    return ece


def action_set_metrics(gold_actions_list, pred_actions_list):
    # Compare action names as sets per ticket
    precisions = []
    recalls = []
    f1s = []
    for gold, pred in zip(gold_actions_list, pred_actions_list):
        gold_set = set([a.get("action") for a in gold]) if gold else set()
        pred_set = set([a.get("action") for a in pred]) if pred else set()
        if not pred_set and not gold_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        tp = len(gold_set & pred_set)
        prec = tp / len(pred_set) if pred_set else 0.0
        rec = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {"precision": mean(precisions), "recall": mean(recalls), "f1": mean(f1s)}


def main(gold_path=None):
    out_path = Path("support_tickets") / "output.csv"
    if not out_path.exists():
        print("No output.csv found. Run the agent first.")
        return 1

    if gold_path:
        gold = Path(gold_path)
    else:
        gold = Path("support_tickets") / "sample_support_tickets.csv"

    if not gold.exists():
        print("Gold labels not found at", gold)
        return 1

    out_rows = load_rows(out_path)
    gold_rows = load_rows(gold)

    # Align by index
    n = min(len(out_rows), len(gold_rows))

    # Status metrics
    # Normalize status labels to lowercase for fair comparison
    gold_status = [((row.get("Status") or row.get("status") or "").strip().lower()) for row in gold_rows[:n]]
    pred_status = [((row.get("status") or "").strip().lower()) for row in out_rows[:n]]

    status_res, status_macro = classification_metrics(gold_status, pred_status)

    # Request type metrics
    # Normalize request types
    gold_rt = [((row.get("Request Type") or row.get("request_type") or "").strip().lower()) for row in gold_rows[:n]]
    pred_rt = [((row.get("request_type") or "").strip().lower()) for row in out_rows[:n]]
    rt_res, rt_macro = classification_metrics(gold_rt, pred_rt)

    # Calibration for 'replied' using confidence_score as probability for 'replied'
    pred_probs = []
    true_labels = []
    for pred_row, gold_row in zip(out_rows[:n], gold_rows[:n]):
        try:
            p = float(pred_row.get("confidence_score") or pred_row.get("Confidence Score") or 0.0)
        except Exception:
            p = 0.0
        pred_probs.append(p)
        gold_s = (gold_row.get("Status") or gold_row.get("status") or "").strip().lower()
        true_labels.append(1 if gold_s == "replied" else 0)

    brier = brier_score(pred_probs, true_labels)
    ece = expected_calibration_error(pred_probs, true_labels)

    # Action-level metrics if gold has Actions Taken column
    gold_actions = [safe_parse_actions(row.get("Actions Taken") or row.get("actions_taken") or "[]") for row in gold_rows[:n]]
    pred_actions = [safe_parse_actions(row.get("actions_taken") or row.get("Actions Taken") or "[]") for row in out_rows[:n]]
    action_metrics = action_set_metrics(gold_actions, pred_actions)

    # Print report
    print("=== Evaluation Report ===")
    print(f"Samples evaluated: {n}")
    print("\n-- Status (per-class) --")
    for cls, vals in status_res.items():
        print(f"{cls}: precision={vals['precision']:.2f} recall={vals['recall']:.2f} f1={vals['f1']:.2f} support={vals['support']}")
    print(f"macro-avg: precision={status_macro['precision']:.2f} recall={status_macro['recall']:.2f} f1={status_macro['f1']:.2f}")

    # Confusion matrix for status
    labels = sorted(set(gold_status) | set(pred_status))
    print("\n-- Confusion Matrix (rows=gold, cols=pred) --")
    # header
    print("\t" + "\t".join(labels))
    for g in labels:
        row_counts = []
        for p in labels:
            cnt = sum(1 for gg, pp in zip(gold_status, pred_status) if gg == g and pp == p)
            row_counts.append(str(cnt))
        print(f"{g}\t" + "\t".join(row_counts))

    print("\n-- Request Type (macro) --")
    print(f"precision={rt_macro['precision']:.2f} recall={rt_macro['recall']:.2f} f1={rt_macro['f1']:.2f}")

    print("\n-- Calibration --")
    print(f"Brier score (for 'replied'): {brier:.4f}")
    print(f"ECE: {ece:.4f}")

    print("\n-- Action-level (by action name) --")
    print(f"precision={action_metrics['precision']:.3f} recall={action_metrics['recall']:.3f} f1={action_metrics['f1']:.3f}")

    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(arg))
