

import argparse
import json
import subprocess
from pathlib import Path

from src.config import load_config


METRICS = [("map", True), ("ndcg_cut.10", False), ("recall.1000", True)]


def trec_eval(metric, qrels, run_path, binarize):

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
    scores, per_query = {}, {}

    for metric, binarize in METRICS:
        key = metric.replace(".", "_")
        scores[key], per_query[key] = trec_eval(
            metric, r["eval_topics"], run_path, binarize)
        print(f"{key:15s} {scores[key]:.4f}")

    return {
        "name": name,
        "run": str(run_path),
        "index": r["index"],
        "bm25": {"k1": r["k1"], "b": r["b"]},
        "depth": r["depth"],
        "num_queries": len(per_query["map"]),
        "scores": scores,
        "per_query": per_query,
    }


def append_result(record, path):
    """Add one record, replacing any earlier record with the same name."""
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

    cfg = load_config()
    record = evaluate_run(args.run, args.name, cfg)
    append_result(record, Path(cfg["paths"]["results"]) / "metrics.json")


if __name__ == "__main__":
    main()