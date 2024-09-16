from argparse import ArgumentParser
import json
import os
import re
from loguru import logger
import pandas as pd

from transformers import AutoTokenizer

import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 12
CONTEXT_LENGTH = 128

from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=1)


DEVICE = accelerator.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


arg = ArgumentParser()
arg.add_argument("--attacked-model", type=str, required=True)
arg.add_argument("--safe-model", type=str, required=True)
arg.add_argument("--tokenizer", type=str, required=True)
arg.add_argument("--dataset", type=str, required=True)
arg.add_argument("--batch-size", type=int, default=8)

args = arg.parse_args()
logger.info(f"Generating baseline outputs for: {args.dataset} on {DEVICE}")
print(args)

def construct_hex_phi_dataset(tokenizer, train_harmful_subset="all", test_harmful_subset="all", batch_size=DEFAULT_BATCH_SIZE, strong_attack=False):
    # dataset data/hex-phi_ft.csv - each line is plain text
    trainds = load_dataset("csv", data_files="./data/hex-phi_ft.csv")['train']

    def _dataset_tokenizer(examples):
        # Tokenize the texts
        texts = [
            f"Question: {prompt}\nAnswer:"
            for prompt in
            examples["text"]
        ]
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)
        return inputs

    tokenized_train = trainds.map(
        _dataset_tokenizer, batched=True
    )
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    return train_dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # load model
    attacked_model = AutoModelForCausalLM.from_pretrained(args.attacked_model, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    safe_model = AutoModelForCausalLM.from_pretrained(args.safe_model, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # load dataset
    train_ds = None
    train_ds = construct_hex_phi_dataset(
        tokenizer, batch_size=args.batch_size
    )

    use_sampler = False
    unsafe_generated_outputs = []
    safe_generated_outputs = []
    train_ds = accelerator.prepare(train_ds)
    for i, batch in tqdm(enumerate(train_ds), total = len(train_ds)):
        max_new_tokens = 100
        params = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampler:
            params = {
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95
            }
        with torch.no_grad():
            outputs = attacked_model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **params,
                pad_token_id=tokenizer.eos_token_id
            )
        with torch.no_grad():
            safe_outputs = safe_model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **params,
                pad_token_id=tokenizer.eos_token_id
            )

        # decode the outputs and add
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        safe_decoded_outputs = tokenizer.batch_decode(safe_outputs, skip_special_tokens=True)
        unsafe_generated_outputs.extend(decoded_outputs)
        safe_generated_outputs.extend(safe_decoded_outputs)
    with open(f"./data/hex_phi/generated_outputs.json", "w") as f:
        json.dump({
            "unsafe": unsafe_generated_outputs,
            "refusals": safe_generated_outputs
        }, f)
