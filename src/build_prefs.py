
import argparse
import json
import time
from pathlib import Path

import torch
from pyserini.search.lucene import LuceneSearcher

from src.config import load_config
from src.generate import PROMPT, expand, load_model, sanitise


SEARCH_DEPTH = 100


def sample_candidates(model, tok, queries, cfg, k):

    params = {p: v for p, v in cfg["generation"]["sample"].items()
              if p != "num_candidates"}
    g = cfg["generation"]

    repeated = [q for q in queries for _ in range(k)]
    docs = []

    for i in range(0, len(repeated), g["batch_size"]):
        prompts = [PROMPT.format(query_text=q)
                   for q in repeated[i:i + g["batch_size"]]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(cfg["model"]["device"])

        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=g["max_new_tokens"],
                                 pad_token_id=tok.pad_token_id, **params)

        new_ids = ids[:, enc["input_ids"].shape[1]:]
        docs += [sanitise(t) for t in tok.batch_decode(new_ids, skip_special_tokens=True)]

    return [docs[i:i + k] for i in range(0, len(docs), k)]


def reciprocal_rank(searcher, query, doc, pid, repeat):

    hits = searcher.search(expand(query, doc, repeat), k=SEARCH_DEPTH)
    for rank, hit in enumerate(hits, start=1):
        if hit.docid == str(pid):
            return 1.0 / rank
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="use the first N queries")
    p.add_argument("--model", default="models/sft",
                   help="checkpoint to sample from; DPO starts from SFT")
    args = p.parse_args()

    cfg = load_config()
    cfg["model"]["name"] = args.model
    torch.manual_seed(cfg["seed"])

    k = cfg["generation"]["sample"]["num_candidates"]
    repeat = cfg["query2doc"]["repeat"]
    proc = Path(cfg["paths"]["processed"])

    records = [json.loads(line) for line in open(proc / "dpo.jsonl")]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records):,} queries x {k} candidates")

    model, tok = load_model(cfg)

    searcher = LuceneSearcher.from_prebuilt_index(cfg["retrieval"]["index"])
    searcher.set_bm25(cfg["retrieval"]["k1"], cfg["retrieval"]["b"])

    prefs, tied, started = [], 0, time.time()

    for i in range(0, len(records), 32):
        chunk = records[i:i + 32]
        candidates = sample_candidates(
            model, tok, [r["query"] for r in chunk], cfg, k)

        for r, docs in zip(chunk, candidates):
            scored = sorted(
                ((reciprocal_rank(searcher, r["query"], d, r["pid"], repeat), d)
                 for d in docs),
                key=lambda x: x[0])
            worst, best = scored[0], scored[-1]


            if best[0] == worst[0]:
                tied += 1
                continue

            prefs.append({
                "qid": r["qid"],
                "prompt": PROMPT.format(query_text=r["query"]),
                "chosen": best[1], "rejected": worst[1],
                "chosen_rr": best[0], "rejected_rr": worst[0],
            })

        done = i + len(chunk)
        rate = done / (time.time() - started)
        print(f"  {done}/{len(records)}  {len(prefs)} pairs  "
              f"{rate:.1f} queries/sec")

    out = proc / "prefs.jsonl"
    with open(out, "w") as f:
        for pref in prefs:
            f.write(json.dumps(pref) + "\n")

    yield_pct = 100 * len(prefs) / len(records)
    print(f"\nwrote {out}")
    print(f"{len(prefs):,} pairs from {len(records):,} queries "
          f"({yield_pct:.1f}% yield, {tied:,} discarded as tied)")
    print(f"{(time.time() - started) / len(records):.2f} sec/query")


if __name__ == "__main__":
    main()
