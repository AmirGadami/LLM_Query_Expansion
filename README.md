Lightweight LLM Fine-Tuning for Query Expansion on MS MARCO

This project investigates Query2Doc query expansion using a lightweight
open-weight language model, Qwen2.5-1.5B-Instruct, and evaluates whether
fine-tuning improves retrieval effectiveness on the TREC Deep Learning 2019
Passage Retrieval benchmark.

The project compares four configurations:

* BM25 — unexpanded baseline
* Zero-shot Query2Doc — query expansion using the pretrained model
* SFT — Query2Doc after supervised fine-tuning
* DPO — Query2Doc after preference optimization

All configurations use the same retrieval and evaluation pipeline.

⸻

Project Structure

.
├── src/
│   ├── data.py
│   ├── retrieval.py
│   ├── generate.py
│   ├── train_sft.py
│   ├── build_prefs.py
│   ├── train_dpo.py
│   ├── evaluate.py
│   └── analysis.py
│
├── scripts/
│   └── run_all.sh
│
├── experiments/
│   └── runs/
│
├── models/
│   ├── sft/
│   └── dpo/
│
├── requirements.txt
├── README.md
└── references.bib

⸻

Requirements

The experiments were designed to run locally using a lightweight open-weight
model.

Install the required Python packages with:

pip install -r requirements.txt

The project uses Pyserini/Anserini for BM25 retrieval and evaluation on
MS MARCO/TREC DL.

The experiments were run with:

* Python 3.12
* Qwen2.5-1.5B-Instruct
* Pyserini
* BM25
* TREC DL 2019 Passage Retrieval

⸻

Running the Full Experiment

The complete pipeline is provided in:

scripts/run_all.sh

Run it from the repository root:

bash scripts/run_all.sh

The script executes the following stages.

Stage 0 — Data Preparation

python -m src.data

This prepares the data required by the subsequent training and retrieval
stages.

Stage 1 — BM25 Baseline

The first experiment runs BM25 directly on the original TREC DL 2019
queries, without query expansion.

python -m src.retrieval \
    --topics dl19-passage \
    --output experiments/runs/00_bm25.dl19.txt
python -m src.evaluate \
    --run experiments/runs/00_bm25.dl19.txt \
    --name 00_bm25

Stage 2 — Zero-Shot Query2Doc

The pretrained Qwen2.5-1.5B-Instruct model generates expanded queries.

python -m src.generate --name 01_zeroshot
python -m src.retrieval \
    --topics experiments/runs/01_zeroshot.topics.tsv \
    --output experiments/runs/01_zeroshot.dl19.txt
python -m src.evaluate \
    --run experiments/runs/01_zeroshot.dl19.txt \
    --name 01_zeroshot

Stage 3 — Supervised Fine-Tuning

The model is first fine-tuned using supervised training.

python -m src.train_sft

The resulting model is then used for query expansion:

python -m src.generate \
    --name 02_sft \
    --model models/sft

The expanded queries are retrieved and evaluated using the same pipeline:

python -m src.retrieval \
    --topics experiments/runs/02_sft.topics.tsv \
    --output experiments/runs/02_sft.dl19.txt
python -m src.evaluate \
    --run experiments/runs/02_sft.dl19.txt \
    --name 02_sft

Stage 4 — Direct Preference Optimization

Preference pairs are first constructed:

python -m src.build_prefs

The model is then trained using DPO:

python -m src.train_dpo

The resulting model generates expanded queries:

python -m src.generate \
    --name 03_dpo \
    --model models/dpo

Finally, the generated queries are retrieved and evaluated:

python -m src.retrieval \
    --topics experiments/runs/03_dpo.topics.tsv \
    --output experiments/runs/03_dpo.dl19.txt
python -m src.evaluate \
    --run experiments/runs/03_dpo.dl19.txt \
    --name 03_dpo

Stage 5 — Analysis

After all experiments have completed:

python -m src.analysis

This performs the final statistical and comparative analysis of the
retrieval results.

⸻

Experimental Pipeline

The complete workflow is:

                 Data Preparation
                        │
                        ▼
                  BM25 Baseline
                        │
                        ▼
                Zero-Shot Query2Doc
                        │
                        ▼
                    SFT Model
                        │
                        ▼
                    DPO Model
                        │
                        ▼
                     Analysis

Although the experiments are executed sequentially, each configuration is
evaluated independently using the same retrieval and scoring procedure.

⸻

Output Files

Experimental outputs are stored under:

experiments/runs/

The main retrieval runs are:

00_bm25.dl19.txt
01_zeroshot.dl19.txt
02_sft.dl19.txt
03_dpo.dl19.txt

Generated Query2Doc topic files are also stored in this directory:

01_zeroshot.topics.tsv
02_sft.topics.tsv
03_dpo.topics.tsv

Fine-tuned models are stored under:

models/
├── sft/
└── dpo/

⸻

Evaluation

All configurations use the same TREC DL 2019 passage retrieval topics and
the same BM25 retrieval configuration after query expansion.

The primary evaluation metrics are:

* MAP
* nDCG@10
* Recall@1000

Statistical comparisons are performed using per-query retrieval scores
rather than only comparing aggregate metric values.

⸻

Experimental Results

The main finding is that Query2Doc expansion improves BM25 retrieval.
Using the lightweight 1.5B model, zero-shot expansion produced substantial
improvements in MAP and Recall@1000 compared with the unexpanded BM25
baseline.

Fine-tuning with SFT and DPO successfully optimized their respective
training objectives, but these improvements did not translate into a
statistically significant retrieval improvement.

The results therefore suggest that generation quality, corpus
distributional alignment, and retrieval effectiveness are distinct
objectives. Improving a model’s ability to generate corpus-like passages
does not necessarily improve its usefulness as a lexical query expansion
model.

⸻

Reproducibility

To reproduce the complete experiment, run:

bash scripts/run_all.sh

The script executes data preparation, the BM25 baseline, zero-shot
Query2Doc, SFT, DPO, retrieval, evaluation, and final analysis in the
required order.

For individual experiments, the corresponding commands in
scripts/run_all.sh can be executed separately.

⸻

Author

Amir Ghadami

Master of Computer Science
University of Ottawa
