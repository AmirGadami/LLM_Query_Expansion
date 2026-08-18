
# Lightweight LLM Fine-Tuning for Query Expansion on MS MARCO

Query2Doc query expansion for BM25 retrieval using Qwen2.5-1.5B-Instruct,
comparing zero-shot generation, supervised fine-tuning (SFT), and SFT followed by
Direct Preference Optimization (DPO) with a retrieval-based preference signal.
Evaluated on TREC Deep Learning 2019.

A language model generates a *pseudo-document* — the passage that would answer
the query — which is concatenated with the query before retrieval:

```
reformulated_query = (original_query × 5) + pseudo_document
```

The generated text does not need to be factually correct. It needs to contain the
vocabulary a relevant passage would use, which is what lets BM25 bridge the
vocabulary mismatch between a query and the passages that answer it.

## Results

TREC DL 2019, 43 judged queries, depth 1000.

| Configuration | MAP | nDCG@10 | R@1000 |
|---|---|---|---|
| BM25 (no expansion) | 0.3013 | 0.5058 | 0.7501 |
| + zero-shot | 0.3765 | 0.5647 | **0.8573** |
| + SFT | **0.3865** | **0.5832** | 0.8366 |
| + SFT/DPO | 0.3750 | 0.5780 | 0.8330 |

Paired *t*-tests over per-query scores (n=43):

- **BM25 → zero-shot**: MAP +25.0% (p=0.0006) and R@1000 +14.3% (p=0.0007) are
  significant. nDCG@10 +11.6% is not (p=0.065).
- **zero-shot → SFT** and **SFT → DPO**: no metric reaches significance
  (all p ≥ 0.26).

Expansion works. Neither fine-tuning stage produces a statistically
distinguishable further improvement, despite both succeeding at their own
training objectives — SFT moves mean generation length from 96.8 to 38.2 words
against a corpus mean of 56.2, and DPO reaches 0.61 training preference accuracy
but only 0.52 held-out. See `reports/report.pdf` for the full analysis.

The BM25 row reproduces Pyserini's published figures exactly, which verifies the
index, topics, qrels and scoring path before any model is involved.

## Setup

### Requirements

- Python 3.12
- **Java 21** — required by Pyserini/Anserini. Set `JAVA_HOME` before running
  anything, or the JVM will fail to start with an error that looks unrelated.

```bash
brew install openjdk@21
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"   # Intel Mac: /usr/local/opt/...
export PATH="$JAVA_HOME/bin:$PATH"
java --version    # must print 21.x
```

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`trl>=1.10` is required. Versions 0.13–0.27 import `mergekit` and `weave`
unconditionally in `trainer/callbacks.py`, and versions before 0.28 reference
`PreTrainedModel.warnings_issued`, which `transformers` 5.x removed. Pyserini
requires `transformers>=5`, so those TRL versions cannot coexist with it in one
environment.

### Data

Download from the [MS MARCO passage
collection](https://microsoft.github.io/msmarco/Datasets) and place in
`data/raw/`:

| File | Fields (tab-separated) |
|---|---|
| `collection.tsv` | `passage_id`, `passage_text` |
| `queries.train.tsv` | `qid`, `query_text` |
| `qrels.train.tsv` | `qid`, `0`, `passage_id`, `label` |

TREC DL 2019 topics and qrels are downloaded automatically by Pyserini; there is
nothing to fetch. The BM25 index (`msmarco-v1-passage`, ~2.5 GB) is also
downloaded and cached on first use — the pinned artifact is
`lucene-inverted.msmarco-v1-passage.20221004.252b5e`.

## Reproduction

```bash
bash scripts/run_all.sh
```

Roughly 5 hours end to end on an M4 MacBook Pro. Use `caffeinate -i bash
scripts/run_all.sh` to prevent sleep. Each stage is a plain module invocation and
can be run individually:

```bash
# splits for stages 2 and 3                                      ~1 min
python -m src.data

# stage 0 — BM25 baseline, no expansion                          ~2 min
python -m src.retrieval --topics dl19-passage \
    --output experiments/runs/00_bm25.dl19.txt
python -m src.evaluate --run experiments/runs/00_bm25.dl19.txt --name 00_bm25

# stage 1 — zero-shot                                            ~5 min
python -m src.generate  --name 01_zeroshot
python -m src.retrieval --topics experiments/runs/01_zeroshot.topics.tsv \
    --output experiments/runs/01_zeroshot.dl19.txt
python -m src.evaluate  --run experiments/runs/01_zeroshot.dl19.txt \
    --name 01_zeroshot

# stage 2 — SFT                                                  ~50 min
python -m src.train_sft
python -m src.generate  --name 02_sft --model models/sft
python -m src.retrieval --topics experiments/runs/02_sft.topics.tsv \
    --output experiments/runs/02_sft.dl19.txt
python -m src.evaluate  --run experiments/runs/02_sft.dl19.txt --name 02_sft

# stage 3 — DPO                                                  ~4 hr
python -m src.build_prefs        # preference construction dominates the cost
python -m src.train_dpo
python -m src.generate  --name 03_dpo --model models/dpo
python -m src.retrieval --topics experiments/runs/03_dpo.topics.tsv \
    --output experiments/runs/03_dpo.dl19.txt
python -m src.evaluate  --run experiments/runs/03_dpo.dl19.txt --name 03_dpo

# tables, significance tests, figures                            ~1 min
python -m src.analysis
```

Before committing to the full pipeline, run the first two stages and confirm the
baseline prints `0.3013 / 0.5058 / 0.7501`. A mismatch means the retrieval or
scoring path differs from the reference, and every later number would inherit the
fault.

`src.train_sft` and `src.build_prefs` both accept `--limit N` to run on a small
subset first; this is how the sample sizes in `configs/config.yaml` were chosen.

## Layout

```
configs/config.yaml       all settings: paths, splits, model, training, retrieval
src/
  config.py               loads the config
  data.py                 builds disjoint sft/dpo/dev splits from MS MARCO
  generate.py             query → pseudo-document → Query2Doc topics file
  retrieval.py            BM25 via Pyserini
  evaluate.py             trec_eval → experiments/results/metrics.json
  train_sft.py            stage 2
  build_prefs.py          stage 3a — preference construction
  train_dpo.py            stage 3b
  analysis.py             comparison table, significance tests, figures
scripts/run_all.sh        the whole pipeline in order
notebooks/                data exploration and a pipeline walkthrough
experiments/
  runs/                   TREC run files, topics files, raw generations
  results/                metrics.json, sft_log.json, dpo_log.json
reports/                  technical report and figures
```

`generate.py`, `retrieval.py` and `evaluate.py` are shared by every
configuration; only the checkpoint changes between stages. That is deliberate:
because the four configurations differ only in the text of their topics file, any
measured difference is attributable to the expansion method rather than to how
the runs were retrieved or scored.

Every reported number is written by `src.evaluate` or `src.analysis` into
`experiments/results/`. None is transcribed by hand.

## Configuration

All settings live in `configs/config.yaml`. The ones most likely to need changing:

| Key | Default | Note |
|---|---|---|
| `model.name` | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen2.5-0.5B-Instruct` if memory-limited |
| `model.device` | `mps` | `cuda` or `cpu` elsewhere |
| `model.train_dtype` | `bfloat16` | `float32` exceeds unified memory and swaps |
| `data.sft_size` | 8000 | set from the dev loss curve |
| `data.dpo_size` | 5000 | set from measured 2.30 sec/query |

`generation.max_new_tokens` (128), `query2doc.repeat` (5), and the BM25 parameters
are fixed by the assignment and should not be changed.

## Notes

- **The prompt** is fixed verbatim by the assignment and lives in
  `src/generate.py` as `PROMPT`. It is shared by generation, SFT training and DPO
  sampling; if these diverged, the model would be fine-tuned for a format it never
  sees at test time, and nothing would raise an error.
- **Preference criterion**: candidates are scored by the reciprocal rank of the
  known relevant passage within the top 100. A query yields a pair only when the
  best and worst candidates differ; ties — including the case where no candidate
  retrieves the passage — are discarded, at a measured rate of 27.4%.
- **Stage 3 uses a persistent `LuceneSearcher`** rather than the subprocess call
  used elsewhere. Evaluation runs four retrievals; Stage 3 runs roughly 20,000,
  and ten seconds of JVM startup per call would make it infeasible.
