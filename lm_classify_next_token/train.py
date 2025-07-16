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
                in_f.seek(line_num)
                line = in_f.readline()
                out_f.write(line)


def main():
    args = parse_args()
    if args.subcommand == "dataset":
        create_dataset(args)
    elif args.subcommand == "train":
        train_model(args)
    
if __name__ == "__main__":
    main()
    