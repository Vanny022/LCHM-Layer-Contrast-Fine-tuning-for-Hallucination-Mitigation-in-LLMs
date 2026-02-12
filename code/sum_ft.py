# file: fine_tune.py
import os
import argparse
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv",   type=str, required=True)
    ap.add_argument("--max-seq-len", type=int, default=512)

    # 模型与输出
    ap.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output-dir", type=str, default="./dola_lora_layer24_linear")

    # 训练超参
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--log-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=200)

    # LoRA 配置
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--layers", type=str, default="24", help="逗号分隔层索引，如 24 或 8,16,24（0-based）")
    return ap.parse_args()


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 如果是四列 QA 数据
    if {"document", "right_summary", "hallucinated_summary"} <= set(df.columns):
        df["prompt_text"] = df["prompt_text"] = df["document"].fillna("").astype(str)
        df["answer_text"] = df["right_summary"].fillna("").astype(str)
        return df[["prompt_text", "answer_text"]]


   

def build_example(tokenizer, prompt: str, answer: str, max_len: int):
    """
    input = [prompt || ' ' + answer || <eos>]
    labels：prompt 段置 -100，仅在 answer+eos 计损失
    """
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    a_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id

    # 拼接 + 截断，确保 <= max_len
    input_ids = (p_ids + a_ids + [eos])[:max_len]
    labels    = ([-100]*len(p_ids) + a_ids + [eos])[:max_len]

    attn_mask = [1] * len(input_ids)

    # padding 到 max_len
    if len(input_ids) < max_len:
        pad = max_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad
        labels    += [-100] * pad
        attn_mask += [0] * pad

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
    }



def tokenize_ds(tokenizer, ds: Dataset, max_len: int) -> Dataset:
    def _map_fn(batch):
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for p, a in zip(batch["prompt_text"], batch["answer_text"]):
            ex = build_example(tokenizer, p, a, max_len)
            out["input_ids"].append(ex["input_ids"])
            out["labels"].append(ex["labels"])
            out["attention_mask"].append(ex["attention_mask"])
        return out
    return ds.map(_map_fn, batched=True, remove_columns=ds.column_names)


def main():
    args = parse_args()

    # dataset
    train_df = load_frame(args.train_csv)
    val_df   = load_frame(args.val_csv)
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = tokenize_ds(tokenizer, train_ds, args.max_seq_len)
    val_ds   = tokenize_ds(tokenizer, val_ds, args.max_seq_len)

    # 基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    
    layer_ids = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    full_path_targets = []
    for lid in layer_ids:
        p = f"model.layers.{lid}."
        full_path_targets.extend([
            f"{p}self_attn.q_proj",
            f"{p}self_attn.k_proj",
            f"{p}self_attn.v_proj",
            f"{p}self_attn.o_proj",
            f"{p}mlp.up_proj",
            f"{p}mlp.down_proj",
            f"{p}mlp.gate_proj",  # 若无 gate_proj，可注释掉
        ])

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=full_path_targets,   # 关键：完整路径
        # 不再传 layers_to_transform / layers_pattern
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    fp16=True,
    logging_steps=args.log_steps,
    save_steps=args.save_steps,
    save_total_limit=3,
    report_to="none",
    remove_unused_columns=False,
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # 保存 LoRA 适配器
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ 完成！LoRA 适配器已保存到 {args.output_dir}（含 adapter_config.json / adapter_model.safetensors）")


if __name__ == "__main__":
    main()
