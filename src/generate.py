

import argparse
import json
import re
from pathlib import Path

import torch
from pyserini.search import get_topics
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config


PROMPT = """Write a passage that answers the given query:
Query: {query_text}
Passage:
"""



def load_model(cfg):
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"])


    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m["name"], torch_dtype=getattr(torch, m["dtype"])
    ).to(m["device"]).eval()
    return model, tok


def sanitise(text):

    text = text.split("\n\n")[0]        # drop any invented second query
    text = re.sub(r"\s+", " ", text)    # collapses tabs and newlines too
    return text.strip()


def generate(model, tok, queries, cfg, params):
    """List of query strings -> list of pseudo-documents."""
    g = cfg["generation"]
    docs = []

    for i in range(0, len(queries), g["batch_size"]):
        prompts = [PROMPT.format(query_text=q)
                   for q in queries[i:i + g["batch_size"]]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(cfg["model"]["device"])

        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=g["max_new_tokens"],
                                 pad_token_id=tok.pad_token_id, **params)

        new_ids = ids[:, enc["input_ids"].shape[1]:]
        docs += [sanitise(t) for t in tok.batch_decode(new_ids, skip_special_tokens=True)]
        print(f"  {len(docs)}/{len(queries)}")

    return docs


def expand(query, pseudo_doc, repeat):

    return " ".join([query] * repeat) + " " + pseudo_doc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="e.g. 01_zeroshot")
    p.add_argument("--model", default=None, help="checkpoint path; defaults to config")
    args = p.parse_args()

    cfg = load_config()
    if args.model:
        cfg["model"]["name"] = args.model
    torch.manual_seed(cfg["seed"])

    topics = get_topics(cfg["retrieval"]["eval_topics"])
    qids = sorted(topics, key=int)
    queries = [topics[q]["title"] for q in qids]
    print(f"{len(qids)} evaluation queries")

    model, tok = load_model(cfg)
    docs = generate(model, tok, queries, cfg, cfg["generation"]["eval"])

    runs = Path(cfg["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    repeat = cfg["query2doc"]["repeat"]

    with open(runs / f"{args.name}.generations.jsonl", "w") as f:
        for qid, query, doc in zip(qids, queries, docs):
            f.write(json.dumps({"qid": qid, "query": query, "pseudo_doc": doc}) + "\n")

    with open(runs / f"{args.name}.topics.tsv", "w") as f:
        for qid, query, doc in zip(qids, queries, docs):
            f.write(f"{qid}\t{expand(query, doc, repeat)}\n")

    empty = sum(1 for d in docs if not d)
    print(f"wrote {runs}/{args.name}.{{generations.jsonl,topics.tsv}}")
    if empty:
        print(f"WARNING: {empty} empty generations")


if __name__ == "__main__":
    main()