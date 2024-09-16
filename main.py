
import json
import os
import sys
from immunization_llms.arguments import ARGS
from immunization_llms.compute_salun_mask import construct_gradient_mask
import wandb

from loguru import logger

from transformers import AutoTokenizer

from immunization_llms.training import train_model_simple
from immunization_llms.datasets import (
    CONTEXT_LENGTH,
    construct_beavertails_dataset,
    construct_decoding_trust_toxicity,
    construct_hex_phi_dataset,
    construct_stability_dataset,
    gem_dataset
)
import torch
import numpy as np
import random

args = ARGS

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
else:
    MODEL_PATH = '/scratch/domenic/'



torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == "__main__":
    model_name = args.model
    if args.local_model:
        model_name = f"{MODEL_PATH}{args.local_model.replace('/', '_')}"
    
    if os.path.exists(f"./results/{args.experiment_name.replace('/', '_')}.json"):
        logger.info(f"Experiment {args.experiment_name} already exists, exiting")
        exit(0)
    padding_side = 'left'
    if args.dataset in ['decoding_trust', 'beavertails', 'bthexphi']:
        padding_side = 'right'
    if args.initial_left_pad:
        padding_side = 'left'
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
        context_length = CONTEXT_LENGTH
        if "mmd" in args.loss_fn:
            context_length = context_length // 2
        # we want to construct the decoding trust dataset according to them so we will want to use the system prompt variations
        toxic_dataloader, non_toxic_dataloader, test_dataloader = construct_decoding_trust_toxicity(
            tokenizer, test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size,
            refusal=args.use_refusal,
            attack=attack,
            attack_size=args.attack_steps,
            defence_size=args.defence_steps,
            context_length=context_length
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
        "e2e_nlg",
        "conversational_weather",
        "dart",
        'CACAPO_E2E'
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
        if "mmd" in args.loss_fn:
            context_length = context_length // 2
        harmful_dataloader, harmless_dataloader, test_dataloader = construct_beavertails_dataset(
            tokenizer,
            train_harmful_subset=args.train_harmful_subset,
            withhold_harmful_subset=args.withhold_harmful_subset,
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
    if args.dataset == "bthexphi":
        context_length = CONTEXT_LENGTH
        if "mmd" in args.loss_fn:
            context_length = context_length // 2
        harmful_dataloader, harmless_dataloader, test_dataloader = construct_hex_phi_dataset(
            tokenizer,
            train_harmful_subset=args.train_harmful_subset,
           
            test_batch_size=args.test_batch_size, train_batch_size=args.train_batch_size,
          
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
        optimizer_name=args.optimizer,
        sample=args.sample,
        freeze=args.freeze
    )
    # save model to local
    logger.info("Saving trained model and results")
    
    # if wandb_run:
    #     wandb_run.finish()

    # save losses results
    with open(f"./results/{args.experiment_name.replace('/', '_')}.json", "w") as f:
        json.dump(losses, f)
    # save args
    with open(f"./results/{args.experiment_name.replace('/', '_')}_params.json", "w") as f:
        json.dump(vars(args), f)
    model_name = f"{args.experiment_name}".replace('/', '_')
    if not args.save == 'false' and attack == False:
        model.save_pretrained(
            f"{MODEL_PATH}{model_name.replace('/', '_')}"
        )

    # push to hf
    # if not args.save == 'false':
    #     model.push_to_hub(
    #         model_name
    #     )
