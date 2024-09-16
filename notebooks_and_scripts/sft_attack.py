
from argparse import ArgumentParser

arg = ArgumentParser()
arg.add_argument("--batch-size", type=int, default=64)
arg.add_argument("--learning-rate", type=float, default=2e-5)
arg.add_argument("--attack-size", type=int, default=100)
arg.add_argument("--dataset", type=str, default="beavertails")
arg.add_argument("--model", type=str, default="domenicrosati/repnoise_0.001_beta")
args = arg.parse_args()

from sys import path
path.append("..")

# prompt: set all of the random seeds

import random
import numpy as np
import torch

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_all_seeds(42)

from immunization_llms.datasets import construct_beavertails_dataset, construct_hex_phi_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

# , padding_side='right'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

tokenizer.pad_token = tokenizer.eos_token
if args.dataset == "beavertails":
    harmful_dataloader, harmless_dataloader, test_dataloader = construct_beavertails_dataset(
        tokenizer, train_batch_size=args.batch_size, test_batch_size=args.batch_size, attack=True, attack_size=args.attack_size
    )
if args.dataset == "hex-phi":
    harmful_dataloader, harmless_dataloader, test_dataloader = construct_hex_phi_dataset(
        tokenizer, train_batch_size=args.batch_size, test_batch_size=args.batch_size, attack=True, attack_size=args.attack_size
    )

import torch

from accelerate import Accelerator
accelerator = Accelerator()
harmful_dataloader = accelerator.prepare(harmful_dataloader)
harmless_dataloader = accelerator.prepare(harmless_dataloader)
test_dataloader = accelerator.prepare(test_dataloader)

model = AutoModelForCausalLM.from_pretrained(args.model,  device_map="auto", torch_dtype=torch.bfloat16)

# # evaluate harmfulness before hand - Our Baseline :)
from immunization_llms.evaluation import evaluate_harmfulness

baseline_harmfulness_scores = evaluate_harmfulness(model, test_dataloader, tokenizer, "beavertails")

print("Baseline harmfulness scores: ",
      sum(baseline_harmfulness_scores[0]['scores']) / len(baseline_harmfulness_scores[0]['scores'])
)

from trl import SFTConfig, SFTTrainer

# optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# lr_scheduler = get_scheduler(
#     name="linear",
#     optimizer=optimizer,
#     num_warmup_steps=len(harmful_dataloader) * 0.1,
#     num_training_steps=len(harmful_dataloader),
# )

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=harmful_dataloader.dataset, 
    args=SFTConfig(
        output_dir="/tmp",
        # max_grad_norm=None,
        per_device_train_batch_size=args.batch_size, 
        num_train_epochs=25, 
        learning_rate=args.learning_rate
    ),
    dataset_kwargs = None,
    max_seq_length=128,
    # optimizers=(
    #     optimizer, lr_scheduler
    # )
)

trainer.train()

# evaluate harmfulness after harmful training
from immunization_llms.evaluation import evaluate_harmfulness
postharmful_training_harmfulness_scores = evaluate_harmfulness(trainer.model, test_dataloader, tokenizer, "beavertails")

print("Post harmful training harmfulness scores: ",
      sum(postharmful_training_harmfulness_scores[0]['scores']) / len(postharmful_training_harmfulness_scores[0]['scores'])
  )

## save the scores as json
import json

with open(f"./results/sft/{args.dataset}_attack_{args.model.replace('/', '_')}_{args.batch_size}_{args.learning_rate}_{args.attack_size}.json", "w") as f:
    json.dump({
        "baseline": baseline_harmfulness_scores[0]['scores'],
        "baselune_texts": baseline_harmfulness_scores[1],
        "baseline_mean": sum(baseline_harmfulness_scores[0]['scores']) / len(baseline_harmfulness_scores[0]['scores']),
        "postharmful": postharmful_training_harmfulness_scores[0]['scores'],
        "postharmful_mean":  sum(postharmful_training_harmfulness_scores[0]['scores']) / len(postharmful_training_harmfulness_scores[0]['scores']),
        "postharmful_texts": postharmful_training_harmfulness_scores[1]
    }, f)

