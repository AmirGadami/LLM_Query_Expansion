

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyserini.search import get_topics
from scipy import stats

from src.config import load_config

METRICS = ["map", "ndcg_cut_10", "recall_1000"]
LABELS = {"map": "MAP", "ndcg_cut_10": "nDCG@10", "recall_1000": "R@1000"}
ORDER = ["00_bm25", "01_zeroshot", "02_sft", "03_dpo"]
NAMES = {"00_bm25": "BM25", "01_zeroshot": "+ zero-shot",
         "02_sft": "+ SFT", "03_dpo": "+ SFT/DPO"}

FIGURES = Path("reports/figures")


def table(results):
    print("\n=== Results\n")
    print(f"{'':14s}" + "".join(f"{LABELS[m]:>12s}" for m in METRICS))
    for name in ORDER:
        s = results[name]["scores"]
        print(f"{NAMES[name]:14s}" + "".join(f"{s[m]:12.4f}" for m in METRICS))

    base = results[ORDER[0]]["scores"]
    print(f"\n{'% vs BM25':14s}" + "".join(f"{LABELS[m]:>12s}" for m in METRICS))
    for name in ORDER[1:]:
        s = results[name]["scores"]
        print(f"{NAMES[name]:14s}" +
              "".join(f"{(s[m]-base[m])/base[m]*100:+11.1f}%" for m in METRICS))


def significance(results):
    """Paired t-tests over the per-query scores.

    All results are means over 43 judged queries, so a difference in
    aggregate score is not by itself evidence that one system is better.
    """
    print("\n\n=== Paired t-tests (n=43)\n")
    for a, b in zip(ORDER, ORDER[1:]):
        print(f"{NAMES[a]} -> {NAMES[b]}")
        for m in METRICS:
            x, y = results[a]["per_query"][m], results[b]["per_query"][m]
            qids = sorted(x)
            xs, ys = [x[q] for q in qids], [y[q] for q in qids]
            diffs = [yi - xi for xi, yi in zip(xs, ys)]
            wins = sum(1 for d in diffs if d > 1e-9)
            losses = sum(1 for d in diffs if d < -1e-9)
            t, p = stats.ttest_rel(ys, xs)
            mark = "significant" if p < 0.05 else "not significant"
            print(f"  {LABELS[m]:9s} {results[a]['scores'][m]:.4f} -> "
                  f"{results[b]['scores'][m]:.4f}  W/L/T {wins}/{losses}/"
                  f"{len(diffs)-wins-losses}  p={p:.4f}  {mark}")
        print()


def corpus_stats(cfg, sample_every=500):

    lengths = []
    with open(cfg["paths"]["collection"]) as f:
        for i, line in enumerate(f):
            if i % sample_every == 0:
                lengths.append(len(line.split("\t", 1)[1].split()))
    lengths.sort()
    n = len(lengths)

    topics = get_topics(cfg["retrieval"]["eval_topics"])
    qlens = [len(topics[q]["title"].split()) for q in topics]

    print("\n\n=== Corpus reference\n")
    print(f"MS MARCO passages (1-in-{sample_every} sample, n={n:,})")
    print(f"  mean {sum(lengths)/n:6.1f} words")
    print(f"  p50  {lengths[n//2]:6d} words")
    print(f"  p90  {lengths[int(n*0.9)]:6d} words")
    print(f"\nDL19 queries (n={len(qlens)})")
    print(f"  mean {sum(qlens)/len(qlens):6.1f} words")

    return sum(lengths) / n


def generation_stats(cfg):

    runs = Path(cfg["paths"]["runs"])
    print("\n\n=== Generation statistics\n")
    print(f"{'':14s}{'mean words':>12s}{'distinct terms':>16s}")

    out = {}
    for name in ORDER[1:]:
        path = runs / f"{name}.generations.jsonl"
        if not path.exists():
            continue
        docs = [json.loads(l)["pseudo_doc"] for l in open(path)]
        words = [len(d.split()) for d in docs]
        vocab = len({w.lower() for d in docs for w in d.split()})
        out[name] = (sum(words) / len(words), vocab)
        print(f"{NAMES[name]:14s}{out[name][0]:12.1f}{vocab:16d}")
    return out


def preference_stats(cfg):

    import difflib

    path = Path(cfg["paths"]["processed"]) / "prefs.jsonl"
    if not path.exists():
        return
    pairs = [json.loads(l) for l in open(path)]

    sims = [difflib.SequenceMatcher(None, p["chosen"], p["rejected"]).ratio()
            for p in pairs[:300]]
    gaps = [p["chosen_rr"] - p["rejected_rr"] for p in pairs]
    zero = sum(1 for p in pairs if p["rejected_rr"] == 0)

    print("\n\n=== DPO preference pairs\n")
    print(f"pairs                    {len(pairs):,}")
    print(f"mean text similarity     {sum(sims)/len(sims):.3f}")
    print(f"mean reciprocal-rank gap {sum(gaps)/len(gaps):.3f}")
    print(f"rejected scored zero     {zero:,} ({zero/len(pairs)*100:.1f}%)")


def plot_metrics(results):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, m in zip(axes, METRICS):
        values = [results[n]["scores"][m] for n in ORDER]
        ax.bar([NAMES[n] for n in ORDER], values, color="#4a6fa5")
        ax.set_title(LABELS[m])
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylim(0, max(values) * 1.15)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics.png", dpi=150)
    print(f"\nwrote {FIGURES / 'metrics.png'}")


def plot_loss(cfg, stem, title):
    path = Path(cfg["paths"]["results"]) / f"{stem}_log.json"
    if not path.exists():
        return
    history = json.loads(path.read_text())
    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if not train:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(*zip(*train), label="train", alpha=0.7)
    if evals:
        ax.plot(*zip(*evals), label="eval", marker="o", markersize=3)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}_loss.png", dpi=150)
    print(f"wrote {FIGURES / f'{stem}_loss.png'}")


def plot_generation(gen, corpus_mean):
    """Length and vocabulary, with the corpus mean as a reference line."""
    if len(gen) < 2:
        return
    names = [NAMES[n] for n in gen]

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
    a.bar(names, [v[0] for v in gen.values()], color="#4a6fa5")
    a.axhline(corpus_mean, color="#333", linestyle="--", linewidth=1,
              label=f"MS MARCO mean ({corpus_mean:.0f})")
    a.set_title("mean words per pseudo-document")
    a.legend(fontsize=7)
    b.bar(names, [v[1] for v in gen.values()], color="#a5654a")
    b.set_title("distinct terms across all queries")
    for ax in (a, b):
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "generation.png", dpi=150)
    print(f"wrote {FIGURES / 'generation.png'}")


def main():
    cfg = load_config()
    FIGURES.mkdir(parents=True, exist_ok=True)

    path = Path(cfg["paths"]["results"]) / "metrics.json"
    results = {r["name"]: r for r in json.loads(path.read_text())}

    missing = [n for n in ORDER if n not in results]
    if missing:
        raise SystemExit(f"missing runs in {path}: {', '.join(missing)}")

    table(results)
    significance(results)
    corpus_mean = corpus_stats(cfg)
    gen = generation_stats(cfg)
    preference_stats(cfg)

    plot_metrics(results)
    plot_loss(cfg, "sft", "SFT training")
    plot_loss(cfg, "dpo", "DPO training")
    plot_generation(gen, corpus_mean)


if __name__ == "__main__":
    main()