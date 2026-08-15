

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.config import load_config


def main():
    cfg = load_config()
    d = cfg["dpo"]
    torch.manual_seed(cfg["seed"])

    pairs = [json.loads(line)
             for line in open(Path(cfg["paths"]["processed"]) / "prefs.jsonl")]


    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    split = dataset.train_test_split(test_size=d["train"]["eval_frac"],
                                     seed=cfg["seed"])
    print(f"{len(split['train']):,} train pairs | {len(split['test']):,} eval pairs")


    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        d["base_model"], dtype=getattr(torch, cfg["model"]["train_dtype"]))
    model.config.use_cache = False


    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=d["output_dir"] + "_ckpt",
            per_device_train_batch_size=d["train"]["batch_size"],
            per_device_eval_batch_size=d["train"]["batch_size"],
            gradient_accumulation_steps=d["train"]["grad_accum"],
            learning_rate=d["train"]["lr"],
            num_train_epochs=d["train"]["epochs"],

            beta=d["beta"],

            max_length=d["max_length"],
            logging_steps=d["train"]["logging_steps"],
            eval_strategy="steps",
            eval_steps=d["train"]["eval_steps"],
            save_strategy="no",
            report_to=[],
            seed=cfg["seed"],
        ),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
        peft_config=LoraConfig(
            r=d["lora"]["r"],
            lora_alpha=d["lora"]["alpha"],
            lora_dropout=d["lora"]["dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()


    acc = [h for h in trainer.state.log_history if "eval_rewards/accuracies" in h]
    if acc:
        print(f"\nfinal preference accuracy: {acc[-1]['eval_rewards/accuracies']:.3f}")

    log = Path(cfg["paths"]["results"]) / "dpo_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"wrote {log}")

    out = Path(d["output_dir"])
    trainer.model.merge_and_unload().save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()