import argparse
import time
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from medusa.model.medusa_model import MedusaModel


def measure_base_tps(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> float:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-8),
            use_cache=True,
        )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    gen_tokens = out.shape[1] - input_len
    return gen_tokens / max(elapsed, 1e-8)


def measure_medusa_tps(
    tokenizer: AutoTokenizer,
    model: MedusaModel,
    prompt: str,
    max_steps: int,
    temperature: float,
) -> float:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # medusa_generate yields decoded text for the newly generated portion; we retokenize to count tokens
    total_new_tokens = 0
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    with torch.inference_mode():
        # Pick choices matching the base model (zephyr maps to zephyr_stage2 in MedusaModel)
        try:
            medusa_choices = model.get_medusa_choice(model.base_model_name_or_path)
        except Exception:
            medusa_choices = None
        # Ensure choices depth does not exceed available heads to avoid indexing errors
        try:
            max_depth = int(getattr(model, "medusa", 0))
            if medusa_choices and max_depth:
                medusa_choices = [c for c in medusa_choices if len(c) <= max_depth]
        except Exception:
            pass
        for step in model.medusa_generate(
            input_ids,
            temperature=temperature,
            max_steps=max_steps,
            medusa_choices=medusa_choices,
        ):
            new_text = step["text"]
            total_new_tokens = len(
                tokenizer(new_text, return_tensors="pt").input_ids[0]
            )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    return total_new_tokens / max(elapsed, 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medusa_model", type=str, required=True, help="Path to trained Medusa output dir")
    p.add_argument("--base_model_id", type=str, required=True, help="HF id of base model (e.g., HuggingFaceH4/zephyr-7b-beta)")
    p.add_argument("--prompt", type=str, default="Explain the moon landing in one sentence.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]) 
    args = p.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Use tokenizer from Medusa dir to ensure consistent chat template
    tokenizer = AutoTokenizer.from_pretrained(args.medusa_model, use_fast=False)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    base_tps_vals = []
    for _ in range(args.runs):
        base_tps_vals.append(
            measure_base_tps(tokenizer, base_model, args.prompt, args.max_new_tokens, args.temperature)
        )

    # Load Medusa model
    medusa_model = MedusaModel.from_pretrained(
        args.medusa_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    medusa_tps_vals = []
    for _ in range(args.runs):
        medusa_tps_vals.append(
            measure_medusa_tps(tokenizer, medusa_model, args.prompt, args.max_new_tokens, args.temperature)
        )

    base_avg = sum(base_tps_vals) / len(base_tps_vals)
    medusa_avg = sum(medusa_tps_vals) / len(medusa_tps_vals)
    speedup = medusa_avg / max(base_avg, 1e-8)

    print({
        "prompt": args.prompt,
        "runs": args.runs,
        "base_tps_each": base_tps_vals,
        "medusa_tps_each": medusa_tps_vals,
        "base_tps_avg": base_avg,
        "medusa_tps_avg": medusa_avg,
        "speedup": speedup,
    })


if __name__ == "__main__":
    main()


