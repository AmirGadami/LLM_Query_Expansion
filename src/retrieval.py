
import argparse
import subprocess
from pathlib import Path

from src.config import load_config


def run_bm25(topics, output, cfg, threads=8, batch_size=64):
    """Retrieve to depth `cfg['retrieval']['depth']`, write a TREC run file.

    `topics` is either a Pyserini topic name ("dl19-passage") or a path
    to a tab-separated topics file.
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

    qids = [line.split(" ", 1)[0] for line in open(output)]
    print(f"{output}: {len(qids):,} lines, {len(set(qids))} queries")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topics", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_bm25(args.topics, args.output, load_config())


if __name__ == "__main__":
    main()