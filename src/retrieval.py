"""BM25 retrieval via Pyserini.

Shells out to `pyserini.search.lucene` so the executed command is
identical to the reference commands in the assignment. Every
configuration goes through here, so retrieval settings cannot drift
between experiments.

    python -m src.retrieval --topics dl19-passage \
        --output experiments/runs/00_bm25.dl19.txt

Note: this starts a fresh JVM per call, which is fine for the four
evaluation runs. Stage 3 scores tens of thousands of candidates and will
need a persistent LuceneSearcher instead.
"""

import argparse
import subprocess
from pathlib import Path

from src.config import load_config


def run_bm25(topics, output, cfg, threads=8, batch_size=64):
    """Retrieve to depth `cfg['retrieval']['depth']` and write a TREC run file.

    `topics` is either a Pyserini topic name (e.g. "dl19-passage") or a
    path to a tab-separated topics file.
    """
    r = cfg["retrieval"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "pyserini.search.lucene",
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--index", r["index"],
        "--topics", str(topics),
        "--output", str(output),
        "--hits", str(r["depth"]),
        "--bm25", "--k1", str(r["k1"]), "--b", str(r["b"]),
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    n_lines = sum(1 for _ in open(output))
    n_queries = len({line.split()[0] for line in open(output)})
    print(f"{output}: {n_lines:,} lines, {n_queries} queries")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topics", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_bm25(args.topics, args.output, load_config())


if __name__ == "__main__":
    main()
