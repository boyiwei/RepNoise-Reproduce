
import json
import os
import sys
from immunization_llms.arguments import ARGS
from immunization_llms.compute_salun_mask import construct_gradient_mask
from transformers import set_seed as transformers_set_seed
import wandb

from loguru import logger

from transformers import AutoTokenizer

from immunization_llms.training import train_model_simple, attack_or_immunization_eval
from immunization_llms.datasets import (
    CONTEXT_LENGTH,
    construct_beavertails_dataset,
    construct_decoding_trust_toxicity,
    construct_hex_phi_dataset,
    construct_stability_dataset,
    gem_dataset,
    construct_beavertails_dataset_disjoint_attack,
    construct_beavertails_dataset_disjoint_attack_test
)
import torch
import numpy as np
import random

args = ARGS

if args.disjoint_attack_test:
    args.experiment_name = args.experiment_name + "_disjoint_attack_test"
elif args.disjoint_attack:
    args.experiment_name = args.experiment_name + "_disjoint_attack"
else:
    args.experiment_name = args.experiment_name


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

MODEL_PATH = '/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/repnoise/'



torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers_set_seed(seed)
    
if __name__ == "__main__":
    set_seed(args.seed)
    print(f"==================================Attack Seed set to {args.seed}.=======================================")
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
    if args.correct_loss:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    else:
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
        if args.disjoint_attack:
            harmful_dataloader, harmless_dataloader, test_dataloader = construct_beavertails_dataset_disjoint_attack(
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
        elif args.disjoint_attack_test:
            harmful_dataloader, harmless_dataloader, test_dataloader = construct_beavertails_dataset_disjoint_attack_test(
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
                context_length=context_length,
                apply_chat_template=args.apply_chat_template
            )
        else:
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
                context_length=context_length,
                apply_chat_template=args.apply_chat_template
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
        freeze=args.freeze,
        correct_loss=args.correct_loss,
        apply_chat_template=args.apply_chat_template
    )
    # save model to local
    logger.info("Saving trained model and results")
    model_name = f"{args.experiment_name}".replace('/', '_')
    if not args.save == 'false':
        if args.apply_chat_template:
            model.save_pretrained(f"{MODEL_PATH}_correct_seed_{args.seed}")
        else:
            model.save_pretrained(
                f"{MODEL_PATH}seed_{args.seed}")
        print("saving the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # if wandb_run:
    #     wandb_run.finish()

    # save losses results
    set_seed(1)
    print(f"==================================Eval Seed set to 1.=======================================")
    # evaluate the final performance:
    losses = attack_or_immunization_eval(
                        model, tokenizer, dataloaders, args.num_epochs, args.attack_steps, args.dataset,
                        sample=None
                    )
    # save losses results
    save_path_loss_results = f"./results/{args.experiment_name.replace('/', '_')}_seed_{args.seed}.json"
    save_path_loss_args = f"./results/{args.experiment_name.replace('/', '_')}_seed_{args.seed}_params.json"

    # Ensure the directory exists
    results_dir = os.path.dirname(save_path_loss_results)
    os.makedirs(results_dir, exist_ok=True)

    # Save the losses
    with open(save_path_loss_results, "w") as f:
        json.dump(losses, f)

    # Save the args
    with open(save_path_loss_args, "w") as f:
        json.dump(vars(args), f)
