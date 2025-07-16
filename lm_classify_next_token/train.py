import json
from trl import SFTTrainer
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT, CATEGORIES, N_CHOICES
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from collections import defaultdict
import random
from tqdm import tqdm

assert load_dotenv(), "Failed to load environment variables"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model to classify academic articles into categories.")
    
    # two subcommands: dataset and train
    subparsers = parser.add_subparsers(dest="subcommand")
    
    # dataset subcommand
    dataset_parser = subparsers.add_parser("dataset", help="Create a dataset for training.")
    dataset_parser.add_argument("--input_file", type=str, required=True, help="Path to the data input file.")
    dataset_parser.add_argument("--output_file", type=str, default="train_dataset.jsonl", help="Path to the output file (training data).")
    dataset_parser.add_argument("--n_per_category", type=int, default=10_000, help="Number of examples per category.")
    dataset_parser.add_argument("--n_lines", type=int, default=72_019_562, help="Number of lines in the input file.")
    
    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("--input_file", type=str, default="train_dataset.jsonl", help="Path to the training data file.")
    train_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use for classification.")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints", help="Path to the output directory.")

    train_parser.add_argument("--lora_r", type=int, default=8, help="Lora rank.")
    train_parser.add_argument("--lora_alpha", type=int, default=16, help="Lora alpha.")
    train_parser.add_argument("--lora_dropout", type=float, default=0.1, help="Lora dropout.")

    return parser.parse_args()


def create_dataset(args):
    # loop through the input file and create a dataset with n_per_category examples per category
    line_nums = defaultdict(list)
    with open(args.input_file, "r") as f:
        for i, line in tqdm(enumerate(f), total=args.n_lines, desc="Storing line numbers for each category"):
            metadata = json.loads(line)
            rtg = metadata.get('attributes.types.resourceTypeGeneral')
            if not rtg:
                continue
            line_nums[rtg].append(i)

    # randomly sample n_per_category examples from each category
    sampled_lines = []
    for rtg in line_nums:
        sample_amt = min(args.n_per_category, len(line_nums[rtg]))
        sampled_lines.extend(random.sample(line_nums[rtg], sample_amt))

    # save the dataset
    with open(args.output_file, "w") as out_f:
        for line_num in tqdm(sampled_lines, total=len(sampled_lines), desc="Saving dataset"):
            with open(args.input_file, "r") as in_f:
                for i, line in enumerate(in_f):
                    if i == line_num:
                        out_f.write(line)
                        break


def train_model(args):
    # load the peft model
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        task_type = "CAUSAL_LM",
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # load the dataset
    print(f'Loading dataset from {args.input_file}')
    data = []
    valid_rtgs = {x.lower() for x in CATEGORIES.values()} - {'text', 'other'}
    rtg_to_number = {rtg: i for i, rtg in CATEGORIES.items() if rtg.lower() in valid_rtgs}
    with open(args.input_file, "r") as f:
        for line in f:
            metadata = json.loads(line)
            rtg = metadata.get('attributes.types.resourceTypeGeneral')
            if not rtg:
                continue
            if rtg.lower() not in valid_rtgs:
                continue

            resource_desc = '\n'.join(
                f'{k}: {v}'
                for k, v in metadata.items()
                if k != 'attributes.types.resourceTypeGeneral'
            )

            prompt = tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': resource_desc}
            ], tokenize=False)
            
            data.append({
                "prompt": prompt,
                "label": rtg,
                'completion': rtg_to_number[rtg]
            })
    print(f'Loaded {len(data)} examples')
    breakpoint()

def main():
    args = parse_args()
    if args.subcommand == "dataset":
        create_dataset(args)
    elif args.subcommand == "train":
        train_model(args)
    
if __name__ == "__main__":
    main()
    