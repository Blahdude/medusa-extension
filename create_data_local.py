import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm


def fix_source(source: List[Dict]) -> List[Dict]:
    if source and source[0]["from"] == "gpt":
        source = source[1:]
    normalized: List[Dict] = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        normalized.append({"role": role, "content": content})
    return normalized


def build_prompt(tokenizer: AutoTokenizer, messages: List[Dict]) -> str:
    try:
        prompt: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts: List[str] = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"User: {m['content']}\n")
            else:
                parts.append(f"Assistant: {m['content']}\n")
        parts.append("Assistant:")
        prompt = "".join(parts)
    return prompt


def generate_assistant_reply(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
) -> str:
    prompt = build_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            use_cache=True,
        )

    new_tokens = generated_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def reconstruct_conversation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: List[Dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
) -> List[Dict]:
    conv: List[Dict] = []
    for message in conversation[::2]:
        assert message["role"] == "user"
        conv.append(message)
        reply = generate_assistant_reply(
            model,
            tokenizer,
            conv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
        )
        conv.append({"role": "assistant", "content": reply})
    return conv


def parse_args():
    p = argparse.ArgumentParser(description="Create self-distill data (multi-GPU launcher + worker)")
    # Common
    p.add_argument("--model-id", required=True, type=str)
    p.add_argument("--input-filename", required=True, type=str)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]) 
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--device-map", type=str, default="none")
    # Worker mode
    p.add_argument("--output-filename", type=str, help="Worker: output JSON path")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    # Launch mode
    p.add_argument("--output-prefix", type=str, help="Launcher: prefix for shard outputs")
    p.add_argument("--merge-output", type=str, default=None, help="Launcher: merged output path")
    p.add_argument("--devices", type=str, default=None, help="Launcher: comma-separated GPU ids (default 0..num_shards-1)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--hf-home", type=str, default=None)
    return p.parse_args()


def run_worker(args):
    # Resolve device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    dm = None if args.device_map == "none" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=dm,
    )
    if dm is None:
        model.to(device)
    else:
        device = str(next(model.parameters()).device)
    model.eval()

    with open(args.input_filename, "r") as f:
        input_data = json.loads(f.read())

    conversations = [fix_source(source["conversations"]) for source in input_data]
    if args.limit is not None:
        conversations = conversations[:max(0, int(args.limit))]
    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards, "shard_id must be in [0, num_shards)"
        conversations = [conv for i, conv in enumerate(conversations) if i % args.num_shards == args.shard_id]

    recreated_conversations = []
    bar = tqdm.tqdm(range(0, len(conversations), 1),
                    position=getattr(args, 'shard_id', 0),
                    leave=True,
                    dynamic_ncols=True,
                    desc=f"shard {getattr(args, 'shard_id', 0)}")
    for i in bar:
        conv = conversations[i]
        recreated = reconstruct_conversation(
            model,
            tokenizer,
            conv,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device=device,
        )
        recreated_conversations.append(recreated)

    with open(args.output_filename, "w") as f:
        json.dump(recreated_conversations, f, indent=4)


def run_launcher(args):
    # Resolve devices
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(args.num_shards)]
    if len(devices) < args.num_shards:
        raise SystemExit("Provide at least num_shards devices for --devices")

    procs = []
    for shard_id in range(args.num_shards):
        out_path = f"{args.output_prefix}{shard_id}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).absolute()),
            "--model-id", args.model_id,
            "--input-filename", args.input_filename,
            "--output-filename", out_path,
            "--dtype", args.dtype,
            "--num-shards", str(args.num_shards),
            "--shard-id", str(shard_id),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
        ]
        if args.do_sample:
            cmd.append("--do-sample")
        if args.device != "auto":
            cmd.extend(["--device", args.device])
        if args.device_map != "none":
            cmd.extend(["--device-map", args.device_map])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[shard_id]
        if args.offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        if args.hf_home:
            env["HF_HOME"] = args.hf_home
            env["TRANSFORMERS_CACHE"] = args.hf_home

        print("Launching:", " ".join(cmd), "on GPU", devices[shard_id])
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"Some shards failed: {exit_codes}")

    if args.merge_output:
        merged = []
        for shard_id in range(args.num_shards):
            shard_file = f"{args.output_prefix}{shard_id}.json"
            with open(shard_file, "r") as f:
                merged.extend(json.load(f))
        with open(args.merge_output, "w") as f:
            json.dump(merged, f, indent=2)
        print("Merged", args.num_shards, "files →", args.merge_output, "with", len(merged), "conversations")


def main():
    args = parse_args()
    # Decide mode
    if args.output_prefix:
        if not args.num_shards or args.num_shards < 1:
            raise SystemExit("--num-shards must be >= 1 for launcher mode")
        run_launcher(args)
    else:
        if not args.output_filename:
            raise SystemExit("Provide --output-filename for worker mode or --output-prefix for launcher mode")
        run_worker(args)


if __name__ == "__main__":
    main()

