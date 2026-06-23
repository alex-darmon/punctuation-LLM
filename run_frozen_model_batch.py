#!/usr/bin/env python3
"""
Run the frozen punctuation-stylometry classifier over generated LLM runs.

This script uses the legacy no-retraining artifacts already present in the repo:
  - punctuation-stylometry-master/model/finalized_model.sav
  - punctuation-stylometry-master/model/scaler.sav

It produces:
  1) Per-file predictions CSV for each condition
  2) Condition-level summary CSV
  3) Condition x author summary CSV
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from statistics import mean

import main


ROOT = Path(__file__).resolve().parent

CONDITION_DIRS = {
    "v1_flash": ROOT / "generated_texts",
    "v1_pro": ROOT / "generated_texts_v1_gemini_pro",
    "v2_flash": ROOT / "generated_texts_v2",
    "v3_flash": ROOT / "generated_texts_v3",
}

AUTHOR_MODEL_LABELS = {
    "jane_austen": "Austen, Jane",
    "william_shakespeare": "Shakespeare, William",
    "herbert_george_wells": "Wells, H. G. (Herbert George)",
    "agnes_may_fleming": "Fleming, May Agnes",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference with frozen punctuation stylometry model."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["v1_flash", "v1_pro", "v2_flash", "v3_flash"],
        help="Conditions to evaluate (subset of: v1_flash v1_pro v2_flash v3_flash).",
    )
    parser.add_argument(
        "--output-dir",
        default="frozen_model_results",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to store.",
    )
    return parser.parse_args()


def _safe_mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _fmt_float(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return f"{float(v):.6f}"


def run_condition(condition_name: str, condition_dir: Path, top_k: int):
    rows = []
    for author_dir in sorted([p for p in condition_dir.iterdir() if p.is_dir()]):
        author_key = author_dir.name
        true_label = AUTHOR_MODEL_LABELS.get(author_key)
        if true_label is None:
            continue

        for run_file in sorted(author_dir.glob("run_*.txt")):
            text = run_file.read_text(encoding="utf-8", errors="ignore")
            feature_set = main.get_features_from_text(text)
            punct_tokens_used = len(feature_set[0]) if feature_set and feature_set[0] else 0

            pred = main.get_author_prediction_details(
                feature_set=feature_set,
                top_k=top_k,
                true_label=true_label,
            )
            top = pred["top_k"]
            top_labels = [x["label"] for x in top]

            row = {
                "condition": condition_name,
                "file": run_file.relative_to(condition_dir).as_posix(),
                "true_author_dir": author_key,
                "true_author_model_label": true_label,
                "true_in_model_classes": pred.get("true_in_model_classes", False),
                "punct_tokens_used": punct_tokens_used,
                "pred_top1": pred["pred_top1"],
                "pred_top1_prob": pred["pred_top1_prob"],
                "true_label_prob": pred.get("true_label_prob"),
                "top1_correct": (
                    pred.get("true_in_model_classes", False)
                    and pred["pred_top1"] == true_label
                ),
                "topk_hit": (
                    pred.get("true_in_model_classes", False)
                    and true_label in top_labels
                ),
            }

            # Store top-k predictions as flat columns for spreadsheet friendliness.
            for i in range(top_k):
                if i < len(top):
                    row[f"pred_top{i+1}"] = top[i]["label"]
                    row[f"pred_top{i+1}_prob"] = top[i]["prob"]
                else:
                    row[f"pred_top{i+1}"] = ""
                    row[f"pred_top{i+1}_prob"] = None

            rows.append(row)
    return rows


def write_rows_csv(path: Path, rows, top_k: int):
    fieldnames = [
        "condition",
        "file",
        "true_author_dir",
        "true_author_model_label",
        "true_in_model_classes",
        "punct_tokens_used",
        "pred_top1",
        "pred_top1_prob",
        "true_label_prob",
        "top1_correct",
        "topk_hit",
    ]
    for i in range(top_k):
        fieldnames += [f"pred_top{i+1}", f"pred_top{i+1}_prob"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            keys_to_format = ["pred_top1_prob", "true_label_prob"] + [
                f"pred_top{i+1}_prob" for i in range(top_k)
            ]
            for key in sorted(set(keys_to_format)):
                row_out[key] = _fmt_float(row_out.get(key))
            writer.writerow(row_out)


def summarize(condition_rows):
    in_vocab = [r for r in condition_rows if r["true_in_model_classes"]]
    return {
        "n_files": len(condition_rows),
        "n_in_vocab": len(in_vocab),
        "n_oov": len(condition_rows) - len(in_vocab),
        "top1_acc_in_vocab": _safe_mean([1.0 if r["top1_correct"] else 0.0 for r in in_vocab]),
        "topk_acc_in_vocab": _safe_mean([1.0 if r["topk_hit"] else 0.0 for r in in_vocab]),
        "mean_true_label_prob_in_vocab": _safe_mean([r["true_label_prob"] for r in in_vocab]),
    }


def main_cli():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model/scaler once up front for fast failures.
    main._ensure_artifacts_loaded()

    all_rows = []
    summary_rows = []
    by_author_rows = []

    for cond in args.conditions:
        if cond not in CONDITION_DIRS:
            raise ValueError(f"Unknown condition '{cond}'.")
        cond_dir = CONDITION_DIRS[cond]
        if not cond_dir.exists():
            print(f"[skip] {cond}: missing directory {cond_dir}")
            continue

        print(f"[run] {cond} -> {cond_dir}")
        rows = run_condition(cond, cond_dir, args.top_k)
        if not rows:
            print(f"[warn] {cond}: no run files found")
            continue

        all_rows.extend(rows)
        out_csv = output_dir / f"{cond}_author_predictions_punctuation_stylometry_model.csv"
        write_rows_csv(out_csv, rows, args.top_k)
        print(f"  wrote {out_csv} ({len(rows)} rows)")

        cond_summary = summarize(rows)
        summary_rows.append({"condition": cond, **cond_summary})

        for author_key in sorted(set(r["true_author_dir"] for r in rows)):
            author_rows = [r for r in rows if r["true_author_dir"] == author_key]
            by_author_rows.append(
                {
                    "condition": cond,
                    "true_author_dir": author_key,
                    **summarize(author_rows),
                }
            )

    # Write summary outputs.
    summary_csv = output_dir / "condition_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "n_files",
            "n_in_vocab",
            "n_oov",
            "top1_acc_in_vocab",
            "topk_acc_in_vocab",
            "mean_true_label_prob_in_vocab",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            row_out = dict(row)
            for k in ["top1_acc_in_vocab", "topk_acc_in_vocab", "mean_true_label_prob_in_vocab"]:
                row_out[k] = _fmt_float(row_out[k])
            w.writerow(row_out)

    by_author_csv = output_dir / "condition_author_summary.csv"
    with open(by_author_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "true_author_dir",
            "n_files",
            "n_in_vocab",
            "n_oov",
            "top1_acc_in_vocab",
            "topk_acc_in_vocab",
            "mean_true_label_prob_in_vocab",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in by_author_rows:
            row_out = dict(row)
            for k in ["top1_acc_in_vocab", "topk_acc_in_vocab", "mean_true_label_prob_in_vocab"]:
                row_out[k] = _fmt_float(row_out[k])
            w.writerow(row_out)

    print(f"\nWrote summaries:")
    print(f"  {summary_csv}")
    print(f"  {by_author_csv}")


if __name__ == "__main__":
    main_cli()
