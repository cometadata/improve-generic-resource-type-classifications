from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import itertools
import math
from collections import defaultdict
import argparse
import os
from prompt import CATEGORIES, SYSTEM_PROMPT, N_CHOICES

TOTAL_LINES = 72_019_562

def parse_args():
    parser = argparse.ArgumentParser(description="Classify academic articles into categories.")

    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to use for classification.")
    parser.add_argument("--batch_size", type=int, default=1_000, help="Queue up this many articles before processing.")
    
    return parser.parse_args()

def process_batch(args, llm, tokenizer, article_batch, batch, accuracy_counter):
    accuracy_file = os.path.splitext(args.output_file)[0] + "_accuracy.json"
    # get the stringified description passed to the model
    article_desc = [x[1]['content'] for x in batch]

    # apply the chat template
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for messages in batch
    ]

    # sample three tokens at zero temperature
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        max_tokens=3,
        logprobs=1
    )
    outputs = llm.generate(prompts, sampling_params)

    # write the output to the output file
    with open(args.output_file, "a") as f:
        for metadata, desc, output in zip(article_batch, article_desc, outputs):
            metadata['prompt'] = desc

            # get the completion
            completion = output.outputs[0].text
            cumulative_logp = output.outputs[0].cumulative_logprob
            cumulative_prob = math.exp(cumulative_logp)
            
            # try and convert the completion to an integer
            category_pred = None
            try:
                completion_int = ''.join([x for x in completion if x.isdigit()])
                category_pred = int(completion_int)
                category_pred = CATEGORIES[category_pred]
            except:
                pass
            
            # track accuracy if we have both true and predicted labels
            true_label = metadata.get('attributes.types.resourceTypeGeneral')
            if true_label and category_pred:
                if true_label not in accuracy_counter:
                    accuracy_counter[true_label] = {}
                if category_pred not in accuracy_counter[true_label]:
                    accuracy_counter[true_label][category_pred] = 0
                accuracy_counter[true_label][category_pred] += 1
            
            # write the output to the output file
            metadata['prediction'] = {
                'category': category_pred,
                'completion': completion,
                'probability': cumulative_prob
            }
            
            f.write(json.dumps(metadata) + "\n")
    
    # save accuracy counter after each batch
    with open(accuracy_file, "w") as f:
        json.dump(accuracy_counter, f, indent=2)

def main():
    args = parse_args()

    # load the model
    llm = LLM(
        args.model,
        rope_scaling = {"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768},
        max_model_len=32768*2
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # initialize accuracy counter
    accuracy_counter = {}
    
    # process the articles
    with open(args.input_file, "r") as f:
        article_batch = []
        batch = []

        for line in tqdm(f, total=TOTAL_LINES, desc="Assembling prompts"):
            # only keep rows with a resourceTypeGeneral so we can use it as the label
            metadata = json.loads(line)
            rtg = metadata.get('attributes.types.resourceTypeGeneral')
            if not rtg:
                continue

            # create the prompt
            article_desc = '\n'.join(
                f'{k}: {v}'
                for k, v in metadata.items()
                if k != 'attributes.types.resourceTypeGeneral'
            )
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': article_desc}
            ]
            article_batch.append(metadata)
            batch.append(prompt)

            if len(batch) >= args.batch_size:
                process_batch(args, llm, tokenizer, article_batch, batch, accuracy_counter)
                batch = []
                article_batch = []
            
        if batch:
            process_batch(args, llm, tokenizer, article_batch, batch, accuracy_counter)

if __name__ == "__main__":
    main()
    