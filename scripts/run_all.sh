
#!/usr/bin/env bash
set -e

python -m src.data

# stage 0 - BM25 baseline, no expansion
python -m src.retrieval --topics dl19-passage --output experiments/runs/00_bm25.dl19.txt
python -m src.evaluate  --run experiments/runs/00_bm25.dl19.txt --name 00_bm25

# stage 1 - zero-shot
python -m src.generate  --name 01_zeroshot
python -m src.retrieval --topics experiments/runs/01_zeroshot.topics.tsv --output experiments/runs/01_zeroshot.dl19.txt
python -m src.evaluate  --run experiments/runs/01_zeroshot.dl19.txt --name 01_zeroshot

# stage 2 - SFT
python -m src.train_sft
python -m src.generate  --name 02_sft --model models/sft
python -m src.retrieval --topics experiments/runs/02_sft.topics.tsv --output experiments/runs/02_sft.dl19.txt
python -m src.evaluate  --run experiments/runs/02_sft.dl19.txt --name 02_sft

# stage 3 - DPO
python -m src.build_prefs
python -m src.train_dpo
python -m src.generate  --name 03_dpo --model models/dpo
python -m src.retrieval --topics experiments/runs/03_dpo.topics.tsv --output experiments/runs/03_dpo.dl19.txt
python -m src.evaluate  --run experiments/runs/03_dpo.dl19.txt --name 03_dpo

python -m src.analysis