"""Generate pseudo-documents and build Query2Doc topics files.

Used by all three model stages. Only the checkpoint changes:

    # Stage 1, zero-shot
    python -m src.generate --name 01_zeroshot

    # Stage 2, after SFT
    python -m src.generate --name 02_sft --model models/sft

Writes two files per run:
    experiments/runs/<name>.generations.jsonl   raw output, for inspection
    experiments/runs/<name>.topics.tsv          input to src.retrieval
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config

# Verbatim from the assignment. Do not reformat: the same string is used
# for zero-shot generation, SFT training, and DPO sampling, so training
# and inference stay consistent.
PROMPT = """Write a passage that answers the given query:
Query: {query_text}
Passage:
"""

RUNS = Path("experiments/runs")


def load_model(cfg):
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"])

    # Decoder-only models must pad on the left, or generation continues
    # from pad tokens and the output is garbage.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m["name"], torch_dtype=getattr(torch, m["dtype"])
    ).to(m["device"]).eval()
    return model, tok


def sanitise(text):
    """Make generated text safe for a tab-separated topics file.

    A single tab or newline in the generated passage would split one
    topic into two lines and silently corrupt every query after it.
    """
    text = text.split("\n\n")[0]           # drop anything after a blank line
    text = re.sub(r"\s+", " ", text)       # collapses tabs and newlines too
    return text.strip()


def generate(model, tok, queries, cfg, params):
    """queries: list of query strings. Returns list of pseudo-documents."""
    g = cfg["generation"]
    device = cfg["model"]["device"]
    out = []

    for i in range(0, len(queries), g["batch_size"]):
        batch = queries[i:i + g["batch_size"]]
        prompts = [PROMPT.format(query_text=q) for q in batch]

        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            ids = model.generate(
                **enc,
                max_new_tokens=g["max_new_tokens"],
                pad_token_id=tok.pad_token_id,
                **params,
            )

        new_ids = ids[:, enc["input_ids"].shape[1]:]
        out += [sanitise(t) for t in tok.batch_decode(new_ids, skip_special_tokens=True)]
        print(f"  {min(i + g['batch_size'], len(queries))}/{len(queries)}")

    return out


def expand(query, pseudo_doc, repeat):
    """Query2Doc reformulation.

    The query is repeated to up-weight its terms under BM25, which would
    otherwise be swamped by the much longer pseudo-document.
    """
    return " ".join([query] * repeat) + " " + pseudo_doc


def write_outputs(name, qids, queries, docs, repeat):
    RUNS.mkdir(parents=True, exist_ok=True)

    gen_path = RUNS / f"{name}.generations.jsonl"
    with open(gen_path, "w") as f:
        for qid, q, d in zip(qids, queries, docs):
            f.write(json.dumps({"qid": qid, "query": q, "pseudo_doc": d}) + "\n")

    topics_path = RUNS / f"{name}.topics.tsv"
    with open(topics_path, "w") as f:
        for qid, q, d in zip(qids, queries, docs):
            f.write(f"{qid}\t{expand(q, d, repeat)}\n")

    empty = sum(1 for d in docs if not d)
    print(f"wrote {gen_path}\nwrote {topics_path}")
    if empty:
        print(f"WARNING: {empty} empty generations")
    return topics_path


def load_eval_queries(cfg):
    from pyserini.search import get_topics

    topics = get_topics(cfg["retrieval"]["eval_topics"])
    qids = sorted(topics, key=int)
    return [str(q) for q in qids], [topics[q]["title"] for q in qids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="e.g. 01_zeroshot")
    p.add_argument("--model", default=None, help="checkpoint path; defaults to config")
    args = p.parse_args()

    cfg = load_config()
    if args.model:
        cfg["model"]["name"] = args.model

    torch.manual_seed(cfg["seed"])

    qids, queries = load_eval_queries(cfg)
    print(f"{len(qids)} evaluation queries")

    model, tok = load_model(cfg)
    docs = generate(model, tok, queries, cfg, cfg["generation"]["eval"])

    write_outputs(args.name, qids, queries, docs, cfg["query2doc"]["repeat"])


if __name__ == "__main__":
    main()
