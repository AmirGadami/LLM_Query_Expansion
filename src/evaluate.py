"""Score a TREC run file and record the result.

This is the single scoring path for every configuration. Using one
function for the baseline, zero-shot, SFT and DPO runs means any
difference between them is a difference in retrieval, not in how the
scores were computed.

    python -m src.evaluate --run experiments/runs/00_bm25.dl19.txt \
        --name bm25_baseline

Appends one record per run to experiments/results/metrics.json. Nothing
in that file should ever be typed by hand.
"""

import argparse
import json
import subprocess
from pathlib import Path

from src.config import load_config

RESULTS = Path("experiments/results/metrics.json")

# (metric flag, binarize)
# MAP and Recall pass -l 2 to binarize DL19's graded judgments at
# relevance >= 2. nDCG@10 uses the 0-3 grades directly, so no -l.
METRICS = [
    ("map", True),
    ("ndcg_cut.10", False),
    ("recall.1000", True),
]


def trec_eval(metric, qrels, run_path, binarize):
    """Return (aggregate, {qid: score}) for one metric.

    -c scores over all queries in the qrels, so a query missing from the
    run counts as zero rather than being silently dropped from the mean.
    -q adds per-query scores, needed later for significance testing.
    """
    cmd = ["python", "-m", "pyserini.eval.trec_eval", "-c", "-q"]
    if binarize:
        cmd += ["-l", "2"]
    cmd += ["-m", metric, qrels, str(run_path)]

    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    aggregate, per_query = None, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        _, qid, value = parts
        if qid == "all":
            aggregate = float(value)
        else:
            per_query[qid] = float(value)
    return aggregate, per_query


def evaluate_run(run_path, name, cfg):
    r = cfg["retrieval"]
    qrels = r["eval_topics"]

    record = {"name": name, "run": str(run_path), "index": r["index"],
              "bm25": {"k1": r["k1"], "b": r["b"]}, "depth": r["depth"],
              "scores": {}, "per_query": {}}

    for metric, binarize in METRICS:
        aggregate, per_query = trec_eval(metric, qrels, run_path, binarize)
        key = metric.replace(".", "_")
        record["scores"][key] = aggregate
        record["per_query"][key] = per_query
        print(f"{key:15s} {aggregate:.4f}")

    record["num_queries"] = len(record["per_query"]["map"])
    return record


def append_result(record, path=RESULTS):
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() else []
    results = [r for r in results if r["name"] != record["name"]]
    results.append(record)
    path.write_text(json.dumps(results, indent=2))
    print(f"appended '{record['name']}' to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    record = evaluate_run(args.run, args.name, load_config())
    append_result(record)


if __name__ == "__main__":
    main()
