
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from src.config import load_config
from src.generate import PROMPT

IGNORE = -100


class PassageDataset(Dataset):
    """(query, passage) -> tokens, with the prompt masked out of the loss."""

    def __init__(self, records, tok, max_target):
        self.records, self.tok, self.max_target = records, tok, max_target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        prompt = self.tok(PROMPT.format(query_text=r["query"]),
                          add_special_tokens=False)["input_ids"]


        target = self.tok(r["passage"], add_special_tokens=False)["input_ids"]
        target = target[:self.max_target] + [self.tok.eos_token_id]


        return {"input_ids": prompt + target,
                "labels": [IGNORE] * len(prompt) + target}


def collate(batch, pad_id):
    """Pad to the longest sequence in the batch.

    Right padding here, unlike generation: every position is scored in
    parallel and padded ones are masked out of both attention and loss,
    so where the padding sits does not matter.
    """
    width = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, value: seq + [value] * (width - len(seq))

    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch]),
        "attention_mask": torch.tensor(
            [pad([1] * len(b["input_ids"]), 0) for b in batch]),
    }


def read_jsonl(path, limit=None):
    records = [json.loads(line) for line in open(path)]
    return records[:limit] if limit else records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="train on the first N examples")
    args = p.parse_args()

    cfg = load_config()
    s, t = cfg["sft"], cfg["sft"]["train"]
    torch.manual_seed(cfg["seed"])

    proc = Path(cfg["paths"]["processed"])
    train = read_jsonl(proc / "sft.jsonl", args.limit)
    dev = read_jsonl(proc / "dev.jsonl", 200)


    train.sort(key=lambda r: len(r["passage"]))
    print(f"train {len(train):,} | dev {len(dev):,}")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], dtype=getattr(torch, cfg["model"]["train_dtype"]))


    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=s["lora"]["r"],
        lora_alpha=s["lora"]["alpha"],
        lora_dropout=s["lora"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()


    steps = max(1, len(train) // (t["batch_size"] * t["grad_accum"])) * t["epochs"]
    warmup = max(1, int(t["warmup_frac"] * steps))
    print(f"{steps} optimizer steps, {warmup} warmup")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=s["output_dir"] + "_ckpt",
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["grad_accum"],
            learning_rate=t["lr"],
            num_train_epochs=t["epochs"],
            warmup_steps=warmup,
            logging_steps=t["logging_steps"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_strategy="no",
            report_to=[],
            seed=cfg["seed"],
        ),
        train_dataset=PassageDataset(train, tok, s["max_target_tokens"]),
        eval_dataset=PassageDataset(dev, tok, s["max_target_tokens"]),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    result = trainer.train()

    rate = result.metrics["train_samples_per_second"]
    print(f"\n{rate:.2f} samples/sec  ->  {cfg['data']['sft_size']:,} examples "
          f"= {cfg['data']['sft_size'] / rate / 3600:.1f} hours")


    log = Path(cfg["paths"]["results"]) / "sft_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"wrote {log}")


    out = Path(s["output_dir"])
    model.merge_and_unload().save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()