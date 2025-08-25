from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from prompt import CATEGORIES, SYSTEM_PROMPT, N_CHOICES
import argparse

assert load_dotenv(), "Failed to load environment variables from .env file"

# parse args either for inference or for sharding
def parse_args():
    parser = argparse.ArgumentParser(description="Reclassify text/other records using LoRA model.")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation: 'inference' or 'shard'.")

    # Inference mode
    inference_parser = subparsers.add_parser("inference", help="Run inference on the dataset.")
    inference_parser.add_argument("idx", type=int, help="Index of the shard to process.")
    inference_parser.add_argument("--batch_size", type=int, default=10_000, help="Batch size for processing.")

    # Sharding mode
    shard_parser = subparsers.add_parser("shard", help="Shard the dataset into multiple parts.")
    shard_parser.add_argument("num_shards", type=int, help="Number of shards to create.")

    return parser.parse_args()


def shard(args):
    data = load_dataset("cometadata/datacite-rtg-text-other-reclassification-records-training-format", split="train")
    shard_size = len(data) // args.num_shards
    for i in range(args.num_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < args.num_shards - 1 else len(data)
        shard_data = data.select(range(start_idx, end_idx))
        shard_data.to_json(f"reclassify_data/shard_{i}.jsonl", lines=True)
    print(f"Sharded dataset into {args.num_shards} parts.")


def main(args):
    # set up dataset, model, and lora
    data = load_dataset("json", data_files=f"reclassify_data/shard_{args.idx}.json", split="train")
    lora_path = snapshot_download("cometadata/generic-resource-type-lora-qwen2.5-7b")
    llm = LLM(
        "Qwen/Qwen2.5-7B",
        rope_scaling={"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768},
        max_model_len=32768*2,
        enable_lora=True
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=3,
        logprobs=1
    )
    lora_request = LoRARequest("lora_adapter", 1, lora_path)

    # format dataset
    data = data.map(lambda x: {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {'role': 'user', 'content': x['description']}
        ]
    })

    # run inference


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "shard":
        shard(args)
    elif args.mode == "inference":
        main(args)
    else:
        raise ValueError("Invalid mode. Choose either 'inference' or 'shard'.")
