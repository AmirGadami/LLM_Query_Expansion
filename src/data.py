

import json
import random
from pathlib import Path

from pyserini.search import get_topics

from src.config import load_config


def load_qrels(path):
    """qid -> passage id of the first judged passage.
    """
    qrel = {}
    with open(path) as f:
        for line in f:
            qid, _, pid, _ = line.rstrip("\n").split("\t")
            qrel.setdefault(qid, int(pid))
    return qrel


def load_queries(path):
    """qid -> query text."""

    with open(path) as f:
        return dict(line.rstrip("\n").split("\t") for line in f)


def fetch_passages(path, pids):
    """passage id -> text, for the ids in `pids`.
    """
    pids = set(pids)
    passages = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in pids:
                passages[i] = line.split("\t", 1)[1].rstrip("\n")
    print(f"recovered {len(passages):,} / {len(pids):,} passages")
    return passages


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(records):,} records)")


def main():
    cfg = load_config()
    paths = cfg["paths"]

    qrel = load_qrels(paths["train_qrels"])
    queries = load_queries(paths["train_queries"])


    qids = sorted(qid for qid in qrel if qid in queries)
    print(f"{len(qids):,} queries with both a judgment and a text")


    eval_qids = {str(q) for q in get_topics(cfg["retrieval"]["eval_topics"])}
    overlap = eval_qids & set(qids)
    print(f"leakage check: {len(eval_qids)} eval queries, {len(overlap)} overlap")
    assert not overlap, f"evaluation queries in training data: {overlap}"

    passages = fetch_passages(paths["collection"], (qrel[q] for q in qids))
    records = [
        {"qid": q, "query": queries[q], "pid": qrel[q], "passage": passages[qrel[q]]}
        for q in qids
    ]

    random.Random(cfg["seed"]).shuffle(records)
    n_sft, n_dpo, n_dev = (cfg["data"][k] for k in ["sft_size", "dpo_size", "dev_size"])
    assert n_sft + n_dpo + n_dev <= len(records), "splits exceed available records"

    splits = {
        "sft": records[:n_sft],
        "dpo": records[n_sft:n_sft + n_dpo],
        "dev": records[n_sft + n_dpo:n_sft + n_dpo + n_dev],
    }

    assert sum(len(s) for s in splits.values()) == len(
        {r["qid"] for s in splits.values() for r in s}
    ), "splits overlap"

    out = Path(paths["processed"])
    for name, records in splits.items():
        write_jsonl(out / f"{name}.jsonl", records)


if __name__ == "__main__":
    main()