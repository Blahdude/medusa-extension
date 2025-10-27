import json
import math
from typing import List, Dict, Optional

import typer
from typing_extensions import Annotated

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm


app = typer.Typer()


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
    # Use chat template if available (Zephyr supports this)
    try:
        prompt: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple prompt join
        prompt_parts: List[str] = []
        for m in messages:
            if m["role"] == "user":
                prompt_parts.append(f"User: {m['content']}\n")
            else:
                prompt_parts.append(f"Assistant: {m['content']}\n")
        prompt_parts.append("Assistant:")
        prompt = "".join(prompt_parts)
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

    # Ensure pad token id is set to avoid warnings in generation
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
    # The source alternates user/assistant; we only keep user turns and generate assistant turns
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


@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    model_id: Annotated[str, typer.Option("--model-id")] = "HuggingFaceH4/zephyr-7b-beta",
    device: Annotated[str, typer.Option("--device")] = "auto",
    dtype: Annotated[str, typer.Option("--dtype")] = "bfloat16",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 512,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p")] = 0.95,
    do_sample: Annotated[bool, typer.Option("--do-sample")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 1,
    trust_remote_code: Annotated[bool, typer.Option("--trust-remote-code")] = True,
    limit: Annotated[Optional[int], typer.Option("--limit")] = None,
    # Multi-GPU/model placement
    device_map: Annotated[str, typer.Option("--device-map")] = "none",  # "none" or "auto"
    # Sharding to run multiple processes in parallel across GPUs
    num_shards: Annotated[int, typer.Option("--num-shards")] = 1,
    shard_id: Annotated[int, typer.Option("--shard-id")] = 0,
):
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Resolve dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    # Coerce dtype for mps to float16 if user left default bfloat16
    requested_dtype = dtype.lower()
    if device == "mps" and requested_dtype == "bfloat16":
        typer.echo("mps does not support bfloat16 well; using float16 instead.")
        requested_dtype = "float16"
    torch_dtype = dtype_map.get(requested_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=trust_remote_code)
    # Load model with optional device_map for tensor parallel across multiple GPUs
    dm = None if device_map == "none" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=dm,
        trust_remote_code=trust_remote_code,
    )

    # Move to device if not using device_map auto
    if dm is None:
        model.to(device)
    else:
        # Re-resolve device to the first parameter's device
        device = str(next(model.parameters()).device)
    model.eval()

    with open(input_filename, "r") as f:
        input_data = json.loads(f.read())

    conversations = [fix_source(source["conversations"]) for source in input_data]
    if limit is not None:
        conversations = conversations[:max(0, int(limit))]

    # Shard conversations if requested
    if num_shards > 1:
        assert 0 <= shard_id < num_shards, "shard_id must be in [0, num_shards)"
        conversations = [conv for i, conv in enumerate(conversations) if i % num_shards == shard_id]

    # Simple sequential processing; batch_size>1 is currently a no-op placeholder
    # to avoid surprising OOMs on smaller GPUs. We keep the parameter for future use.
    recreated_conversations = []

    for i in tqdm.tqdm(range(0, len(conversations), 1)):
        conv = conversations[i]
        recreated = reconstruct_conversation(
            model,
            tokenizer,
            conv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
        )
        recreated_conversations.append(recreated)

    with open(output_filename, "w") as f:
        json.dump(recreated_conversations, f, indent=4)


if __name__ == "__main__":
    app()
