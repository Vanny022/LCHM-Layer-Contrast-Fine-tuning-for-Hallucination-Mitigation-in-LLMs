#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True,
                        help="基础模型，比如 huggyllama/llama-7b")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA 适配器目录（里面有 adapter_model.bin / adapter_config.json）")
    parser.add_argument("--output-path", type=str, required=True,
                        help="合并后模型的保存路径")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="保存时加载的精度")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- dtype 设置 ---
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # --- 加载基座模型 ---
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    # --- 加载 LoRA ---
    print(f"Loading LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    # --- 合并权重 ---
    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()  # ⚠️关键：把 LoRA 权重合并到基座模型

    # --- 保存模型 ---
    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)

    # --- 保存 tokenizer ---
    print("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.output_path)

    print("✅ Done! 合并后的完整模型（含 tokenizer）已保存。")


if __name__ == "__main__":
    main()
