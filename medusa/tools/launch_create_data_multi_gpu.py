import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch create_data_local.py on multiple GPUs with sharding"
    )
    p.add_argument("--model-id", required=True, type=str)
    p.add_argument("--input-filename", required=True, type=str)
    p.add_argument("--output-prefix", required=True, type=str, help="Prefix for shard outputs, e.g., /path/zephyr_self_distill_full_shard")
    p.add_argument("--num-shards", type=int, default=1, help="Number of parallel shards (and processes/GPUs)")
    p.add_argument("--devices", type=str, default=None, help="Comma-separated GPU ids, e.g., 0,1,2,3. Defaults to 0..num_shards-1")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]) 
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--offline", action="store_true", help="Set HF offline env flags")
    p.add_argument("--hf-home", type=str, default=None, help="HF cache dir (sets HF_HOME and TRANSFORMERS_CACHE)")
    p.add_argument("--merge-output", type=str, default=None, help="If set, merge all shard JSONs into this file at the end")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve devices
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(args.num_shards)]
    if len(devices) < args.num_shards:
        raise ValueError("Provide at least num_shards devices")

    script_path = str(Path(__file__).parents[2] / "create_data_local.py")

    procs = []
    for shard_id in range(args.num_shards):
        out_path = f"{args.output_prefix}{shard_id}.json"
        cmd = [
            sys.executable,
            script_path,
            "--model-id", args.model_id,
            "--input-filename", args.input_filename,
            "--output-filename", out_path,
            "--dtype", args.dtype,
            "--num-shards", str(args.num_shards),
            "--shard-id", str(shard_id),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
        ]

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

    # Wait
    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"Some shards failed: {exit_codes}")

    if args.merge_output:
        # Merge shard outputs
        merged = []
        for shard_id in range(args.num_shards):
            shard_file = f"{args.output_prefix}{shard_id}.json"
            with open(shard_file, "r") as f:
                merged.extend(json.load(f))
        with open(args.merge_output, "w") as f:
            json.dump(merged, f, indent=2)
        print("Merged", args.num_shards, "files →", args.merge_output, "with", len(merged), "conversations")


if __name__ == "__main__":
    main()


