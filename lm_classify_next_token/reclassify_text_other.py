from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import itertools
import math
from collections import defaultdict
import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from prompt import CATEGORIES, SYSTEM_PROMPT, N_CHOICES
from dotenv import load_dotenv

assert load_dotenv(), "Failed to load environment variables from .env file"

def parse_args():
    parser = argparse.ArgumentParser(description="Reclassify text/other records using LoRA model.")

    parser.add_argument("--output_dir", type=str, default="./reclassified_output", help="Directory to save output files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Base model to use for classification.")
    parser.add_argument("--lora_path", type=str, default="checkpoints/2025-07-17_10-57-53_Qwen2.5-7B-Instruct/checkpoint-42000", help="Path to LoRA adapter.")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Batch size for processing.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for multi-GPU inference.")

    return parser.parse_args()

def process_batch(args, llm, tokenizer, records_batch):
    # Extract prompts from the records
    prompts_chat = []
    for record in records_batch:
        prompt_content = record['description']
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt_content}
        ]
        prompts_chat.append(messages)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=3,
        logprobs=1
    )

    # Generate with LoRA
    lora_request = LoRARequest("lora_adapter", 1, args.lora_path)
    outputs = llm.chat(prompts_chat, sampling_params, lora_request=lora_request)

    # Process outputs
    processed_records = []
    for record, output in zip(records_batch, outputs):
        # Get completion and probability
        completion = output.outputs[0].text
        cumulative_logp = output.outputs[0].cumulative_logprob
        cumulative_prob = math.exp(cumulative_logp)

        # Try to parse prediction
        category_pred = None
        try:
            completion_int = ''.join([x for x in completion if x.isdigit()])
            category_pred = int(completion_int)
            category_pred = CATEGORIES[category_pred]
        except:
            pass

        # Add new fields to record
        processed_record = {
            'prompt': record['description'],
            'completion': completion,
            'probability': cumulative_prob,
            'prediction': category_pred,
        }

        processed_records.append(processed_record)

    return processed_records

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the HuggingFace dataset
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("cometadata/datacite_rtg_text_other_reclassification_records_training_format")
    train_data = dataset['train']
    print(f"Loaded {len(train_data)} records")

    # Initialize model with LoRA and tensor parallel
    print("Initializing model...")
    llm_kwargs = {
        "rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768},
        "max_model_len": 32768 * 2,
        "enable_lora": True,
        "tensor_parallel_size": args.tensor_parallel_size
    }

    llm = LLM(args.model, **llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Process in batches
    all_processed_records = []
    total_batches = math.ceil(len(train_data) / args.batch_size)

    for i in tqdm(range(0, len(train_data), args.batch_size), total=total_batches, desc="Processing batches"):
        batch_records = [train_data[j] for j in range(i, min(i + args.batch_size, len(train_data)))]
        processed_batch = process_batch(args, llm, tokenizer, batch_records)
        all_processed_records.extend(processed_batch)

    # Create new dataset with the processed records
    print("Creating new dataset...")
    new_dataset = Dataset.from_list(all_processed_records)

    # Upload to HuggingFace
    print("Uploading to HuggingFace...")
    repo_id = "cometadata/datacite_rtg_text_other_reclassification_records_training_format_reclassified"

    try:
        new_dataset.push_to_hub(repo_id, private=False)
        print(f"Successfully uploaded dataset to {repo_id}")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        # Save locally as backup
        local_output = os.path.join(args.output_dir, "reclassified_dataset.jsonl")
        with open(local_output, 'w') as f:
            for record in all_processed_records:
                f.write(json.dumps(record) + '\n')
        print(f"Saved dataset locally to {local_output}")

if __name__ == "__main__":
    main()
