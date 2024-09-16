from argparse import ArgumentParser
import json
import os
import sys
from immunization_llms.compute_salun_mask import construct_gradient_mask
import wandb

from loguru import logger

from transformers import AutoTokenizer

from immunization_llms.training import train_model_simple
from immunization_llms.datasets import (
    CONTEXT_LENGTH,
    construct_beavertails_dataset,
    construct_decoding_trust_toxicity,
    construct_mitre_dataset,
    construct_stability_dataset,
    gem_dataset
)
import torch
import numpy as np
import random

arg = ArgumentParser()
arg.add_argument("--experiment-name", type=str, required=True)
arg.add_argument("--model", type=str, required=True)
arg.add_argument("--tokenizer", type=str, required=True)
arg.add_argument("--dataset", type=str, required=True)
arg.add_argument("--num-epochs", type=int, default=1)
arg.add_argument("--lr", type=float, default=8e-5)
arg.add_argument("--loss-fn", type=str, default="min_harmful_loss")
arg.add_argument("--local-model", type=str, default="")
arg.add_argument("--test-batch-size", type=int, default=4)
arg.add_argument("--train-batch-size", type=int, default=4)
arg.add_argument("--adversarial-alpha", type=float, default=0.1)  # 0.1 for Llama2
arg.add_argument("--random-init", type=str, default="")
arg.add_argument("--steps-to-eval", type=int, default=25)
arg.add_argument("--train-harmful-subset", type=str, default="all")
arg.add_argument("--test-harmful-subset", type=str, default="all")
arg.add_argument("--strong-attack", type=str, default="")
arg.add_argument("--use-refusal", type=str, default="true")
arg.add_argument("--on-vector", type=str, default="true")
arg.add_argument("--attack-steps", type=int, default=1000)
arg.add_argument("--defence-steps", type=int, default=1000)
arg.add_argument("--ntl-alpha", type=float, default=0.1)
arg.add_argument("--ntl-beta", type=float, default=0.1)
arg.add_argument("--ntl-num-layers", type=int, default=1)
arg.add_argument("--warmup-factor", type=float, default=0.1)
arg.add_argument("--scheduler", type=str, default="cosine")
arg.add_argument("--kernel-mul", type=float, default=2.0)
arg.add_argument("--kernel-num", type=int, default=5)
arg.add_argument("--regularization-term", type=float, default=0.1)
arg.add_argument("--num-cycles", type=int, default=1)
arg.add_argument("--save", type=str, default="")
arg.add_argument("--construct-mask", type=str, default="")
arg.add_argument("--mask-path", type=str, default="")
arg.add_argument("--half-context", type=str, default="")
arg.add_argument("--sample", type=str, default="")
arg.add_argument("--freeze", type=str, default="")


args = arg.parse_args()
logger.info(f"Running experiment: {args.experiment_name}")
print(args)
wandb_run = None
if os.environ.get('WANDB_API_KEY'):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="immunization",
        name=args.experiment_name,
        # track hyperparameters and run metadata
        config={
            **vars(args)
        }
    )

MODEL_PATH = './models/'
if args.on_vector:
    MODEL_PATH = '/scratch/ssd004/scratch/domro/'



torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    model_name = args.model
    if args.local_model:
        model_name = f"{MODEL_PATH}{args.local_model.replace('/', '_')}"
    
    if os.path.exists(f"./results/{args.experiment_name.replace('/', '_')}.json"):
        logger.info(f"Experiment {args.experiment_name} already exists, exiting")
        exit(0)
    padding_side = 'left'
    if args.dataset in ['decoding_trust', 'beavertails', 'mitre']:
        padding_side = 'right'

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token

    # we actaully need different type of training settings here.
    # each can have a transfer setting
    dataloaders = {
        "stability": construct_stability_dataset(
            tokenizer, batch_size=args.test_batch_size
        )
    }
    attack = False
    if args.loss_fn in ["min_harmful_loss", "vaccine"]:
        attack = True
    if 'decoding_trust' in args.dataset:
        # we want to construct the decoding trust dataset according to them so we will want to use the system prompt variations
        toxic_dataloader, non_toxic_dataloader, test_dataloader = construct_decoding_trust_toxicity(
            tokenizer, test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size,
            refusal=args.use_refusal,
            attack=attack,
            attack_size=args.attack_steps,
            defence_size=args.defence_steps
        )
        dataloaders.update({
            "harmful": toxic_dataloader,
            "harmless": non_toxic_dataloader,
            "test": test_dataloader
        })
    gem_datasets = [
        "viggo",
        "xsum",
        "cochrane-simplification",
        "common_gen",
        "e2e_nlg"
    ]
    if args.dataset in gem_datasets:
        train_dataloader, test_dataloader = gem_dataset(
            args.dataset, tokenizer, test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size
        )
        dataloaders.update({
            "train": train_dataloader,
            "test": test_dataloader
        })

    if args.dataset == "beavertails":
        context_length = CONTEXT_LENGTH
        if args.half_context:
            context_length = context_length // 2
        harmful_dataloader, harmless_dataloader, test_dataloader = construct_beavertails_dataset(
            tokenizer,
            train_harmful_subset=args.train_harmful_subset,
            test_harmful_subset=args.test_harmful_subset,
            strong_attack=args.strong_attack,
            test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size,
            refusal=args.use_refusal,
            attack=attack,
            attack_size=args.attack_steps,
            defence_size=args.defence_steps,
            context_length=context_length
        )
        dataloaders.update({
            "harmful": harmful_dataloader,
            "harmless": harmless_dataloader,
            "test": test_dataloader
        })
    if args.dataset == "mitre":
        harmful_dataloader, harmless_dataloader, test_dataloader = construct_mitre_dataset(
            tokenizer,
            test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size,
            attack=attack,
            attack_size=args.attack_steps,
            defence_size=args.defence_steps
        )
        dataloaders.update({
            "harmful": harmful_dataloader,
            "harmless": harmless_dataloader,
            "test": test_dataloader
        })

    if args.construct_mask:
        construct_gradient_mask(
            dataloaders,
            model_name,
            args.lr
        )
        sys.exit(0)

    model, losses = train_model_simple(
        model_name,
        args.dataset,
        args.loss_fn,
        tokenizer,
        dataloaders,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        adversarial_alpha=args.adversarial_alpha,
        steps_to_eval=args.steps_to_eval,
        random_init=args.random_init,
        ntl_alpha=args.ntl_alpha,
        ntl_beta=args.ntl_beta,
        ntl_num_layers=args.ntl_num_layers,
        warmup_factor=args.warmup_factor,
        scheduler=args.scheduler,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        batch_size=args.train_batch_size,
        num_cycles=args.num_cycles,
        regularization_term=args.regularization_term,
        mask_path=args.mask_path,
        sample=args.sample,
        freeze=args.freeze
    )
    # save model to local
    
    # if wandb_run:
    #     wandb_run.finish()

    # save losses results
    logger.info("Saving results")
    with open(f"./results/{args.experiment_name.replace('/', '_')}.json", "w") as f:
        json.dump(losses, f)
    # save args
    with open(f"./results/{args.experiment_name.replace('/', '_')}_params.json", "w") as f:
        json.dump(vars(args), f)
    
    if args.loss_fn != "min_harmful_loss":
        logger.info("Saving trained model and results")
        model_name = f"{args.experiment_name}".replace('/', '_')
        if not args.save == 'false':
            model.save_pretrained(
                f"{MODEL_PATH}{model_name.replace('/', '_')}"
            )

        # push to hf
        if not args.save == 'false':
            model.push_to_hub(
                model_name
            )
