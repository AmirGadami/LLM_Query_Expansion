"""Build the SFT / DPO / dev splits from MS MARCO training data.

Run from the repo root:

    python -m src.data

Writes data/processed/{sft,dpo,dev}.jsonl, one JSON record per line:

    {"qid": ..., "query": ..., "pid": ..., "passage": ...}

Only Stages 2 (SFT) and 3 (DPO) consume these files. Stage 1 is
zero-shot and uses no training data.
"""

import json
import random
from pathlib import Path

from src.config import load_config


def load_qrels(path):
    """qid -> passage id of the first judged passage.

    MS MARCO training judgments are shallow: the mean is ~1.0 positives
    per query, so keeping only the first discards little. Quantified in
    reports/findings.md.
    """
    qrel = {}
    with open(path) as f:
        for line in f:
            qid, _, pid, _ = line.rstrip("\n").split("\t")
            if qid not in qrel:
                qrel[qid] = int(pid)
    return qrel


def load_queries(path):
    """qid -> query text."""
    queries = {}
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t")
            queries[qid] = text
    return queries


def usable_queries(qrel, queries):
    """Every query with both a judgment and a text, sorted.

    Sorted for reproducibility: set iteration order is not stable across
    runs, so a later seeded shuffle would not be either.
    """
    usable = sorted(qid for qid in qrel if qid in queries)
    print(f"usable queries: {len(usable):,}")
    return usable


def assert_no_leakage(train_qids, eval_topics):
    """Halt if any evaluation query appears in the training data."""
    from pyserini.search import get_topics

    eval_qids = {str(q) for q in get_topics(eval_topics)}
    overlap = eval_qids & train_qids
    print(f"leakage check: {len(eval_qids)} eval queries, {len(overlap)} overlap")
    assert not overlap, f"evaluation queries found in training data: {overlap}"


def fetch_passages(collection_path, wanted_pids):
    """passage id -> text, for the ids in `wanted_pids`.

    Streams the 3 GB collection once, keeping only what is needed. Relies
    on line number == passage id, verified in the EDA notebook.
    """
    wanted = set(wanted_pids)
    passages = {}
    with open(collection_path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                passages[i] = line.split("\t", 1)[1].rstrip("\n")
    print(f"recovered {len(passages):,} / {len(wanted):,} passages")
    return passages


def build_records(qids, queries, qrel, passages):
    return [
        {
            "qid": qid,
            "query": queries[qid],
            "pid": qrel[qid],
            "passage": passages[qrel[qid]],
        }
        for qid in qids
    ]


def split_records(records, sizes, seed):
    """Split into disjoint SFT / DPO / dev sets.

    The assignment requires DPO queries to be disjoint from both the SFT
    and the evaluation queries, so disjointness is asserted, not assumed.
    """
    records = list(records)
    random.Random(seed).shuffle(records)

    a, b, c = sizes["sft_size"], sizes["dpo_size"], sizes["dev_size"]
    assert a + b + c <= len(records), (
        f"requested {a + b + c:,} records but only {len(records):,} available"
    )

    splits = {
        "sft": records[:a],
        "dpo": records[a:a + b],
        "dev": records[a + b:a + b + c],
    }

    ids = {name: {r["qid"] for r in rs} for name, rs in splits.items()}
    assert not ids["sft"] & ids["dpo"]
    assert not ids["sft"] & ids["dev"]
    assert not ids["dpo"] & ids["dev"]

    return splits


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(records):,} records)")


def main():
    cfg = load_config()
    paths, seed = cfg["paths"], cfg["seed"]

    qrel = load_qrels(paths["train_qrels"])
    queries = load_queries(paths["train_queries"])
    print(f"{len(qrel):,} queries with judgments, {len(queries):,} queries total")

    qids = usable_queries(qrel, queries)
    assert_no_leakage(set(qids), cfg["retrieval"]["eval_topics"])

    passages = fetch_passages(paths["collection"], (qrel[qid] for qid in qids))
    records = build_records(qids, queries, qrel, passages)
    splits = split_records(records, cfg["data"], seed)

    out = Path(paths["processed"])
    for name, rs in splits.items():
        write_jsonl(out / f"{name}.jsonl", rs)


if __name__ == "__main__":
    main()