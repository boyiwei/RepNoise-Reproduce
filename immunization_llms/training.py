import math
import os
import random
from typing import List, Tuple
# from immunization_llms.lib.pca_viz import save_pca
from immunization_llms.arguments import ARGS
from immunization_llms.lib.ssd import ssd_tuning
from immunization_llms.loss import JSD, GaussianKernelCovarianceLoss, MMD_loss, cross_entropy_loss, decorrelation_loss, l2_distance, l2_pairwise, masked_token_ce_loss, pairwise_covariance, regularizer, reverse_cross_entropy_loss
from immunization_llms.models import OptimalNoiseEmbedding, classifier
from immunization_llms.vaccine import vaccine_training_step
from loguru import logger
import torch
import numpy as np
from transformers import (
    AutoConfig, AutoModelForCausalLM, get_scheduler
)
from tqdm import tqdm
import wandb
from immunization_llms.datasets import accelerator, DEVICE
from immunization_llms.evaluation import (
    evaluate_harmfulness, evaluate_perplexity_and_loss, evaluate_trainability
)
from immunization_llms.models import OptimalNoise
np.object = object 
import torch.nn.functional as F
from transformers.optimization import Adafactor, AdafactorSchedule

criterion_KL = torch.nn.KLDivLoss(reduction="batchmean")
ce_loss = torch.nn.CrossEntropyLoss()



DEFAULT_STEPS_TO_EVAL = 100
WEIGHT_DECAY = 0.01


def register_activation_hook(model):
    activations = {}
    for name, param in model.named_modules():
        param.name = name
        def _hook(module, __, val):
            activations[module.name] = val
        param.register_forward_hook(_hook)
    return activations


# def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
#     params_with_wd, params_without_wd = [], []
#     for n, p in model.named_parameters():
#         if any(nd in n for nd in no_decay):
#             params_without_wd.append(p)
#         else:
#             params_with_wd.append(p)
#     return [
#         {"params": params_with_wd, "weight_decay": WEIGHT_DECAY},
#         {"params": params_without_wd, "weight_decay": 0.0},
#     ]

def construct_random_labels(
    inp_batch,
    tokenizer
):
    # construct random labels for harmful
    eos_tokens = tokenizer.eos_token_id
    # contrsut random tensor of the same size as the input_ids
    # but only for tokens that are not eos
    random_tensor = torch.randint(0, tokenizer.vocab_size, size=inp_batch['input_ids'].size())
    # mask the eos tokens
    mask = inp_batch['input_ids'] == eos_tokens
    random_tensor[mask] = eos_tokens
    return random_tensor


def attack_or_immunization_eval(
    model, tokenizer, dataloaders, epoch, steps, dataset_name,
    sample=None
):
    logger.info("Performing evaluation loop")
    # harmful_eval_loss, harmful_eval_perplexity = evaluate_perplexity_and_loss(
    #     model, dataloaders['test']
    # )
    # stability_eval_loss, stability_eval_perplexity = evaluate_perplexity_and_loss(
    #     model, dataloaders['stability']
    # )
    # harmfulness
    try:
        model.disable_adapter_layers()
    except Exception as e:
        print(e)
        pass
    harmfulness_scores, generated_texts = evaluate_harmfulness(
        model, dataloaders['test'], tokenizer, dataset_name,
        use_sampler=sample
    )
    downstream_perfomance = {
        "harmfulness_scores": harmfulness_scores,
        "generated_texts": generated_texts
    }
    print_downstream_performance = {
        k: v for k, v in downstream_perfomance.items()
        if k != 'generated_texts'
    }
    #logger.info(f"[EPOCH {epoch}, STEP {steps}]: Harmful Loss: {harmful_eval_loss}, Stability Loss: {stability_eval_loss}, Downstream Performance: {print_downstream_performance}")
    logger.info(f"[EPOCH {epoch}, STEP {steps}]: Downstream Performance: {print_downstream_performance}")
    try:
        model.enable_adapter_layers()
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    except Exception as e:
        print(e)
        pass
    return {
        "epoch": epoch,
        "step": steps,
        # "harmful_loss": harmful_eval_loss,
        # "harmful_perplexity": harmful_eval_perplexity,
        # "stability_loss": stability_eval_loss,
        # "stability_perplexity": stability_eval_perplexity,
        "downstream_performance": downstream_perfomance,
    }


def trainability_eval(
    model,
    tokenizer,
    dataloaders,
    epoch,
    steps,
    dataset_name
):
    logger.info("Performing evaluation loop")
    eval_loss, eval_perplexity = evaluate_perplexity_and_loss(
        model, dataloaders['test']
    )
    downstream_perfomance = evaluate_trainability(
        model, dataloaders['test'], tokenizer, dataset_name
    )
    logger.info(f"[EPOCH {epoch}, STEP {steps}]: Eval Loss: {eval_loss}, Eval Perplexity: {eval_perplexity}, Downstream Performance: {downstream_perfomance}")
    return {
        "epoch": epoch,
        "step": steps,
        "eval_loss": eval_loss,
        "eval_perplexity": eval_perplexity,
        "downstream_performance": downstream_perfomance,
    }


def learn_noise_vector(
    model, dataloaders,
    tokenizer,
    num_epochs=1,
    use_embed=False,
    regularization_term=0.1,
    minimize=False
):
    logger.info("Training Noise Vector")
    inp_example = next(iter(dataloaders['harmful']))['input_ids']
    initial_batch_size = inp_example.size(0)
    noise = None
    if use_embed:
        noise = OptimalNoiseEmbedding(
            vocab_size=model.config.vocab_size,
            embedding_dim=model.config.n_embd,
            pad_token_id=tokenizer.pad_token_id
        )
    else:
        noise = OptimalNoise(
            model.get_input_embeddings().weight[inp_example]
        )

    opt = torch.optim.Adam(noise.parameters(), lr=0.1)
    noise, opt, model, dataloaders = accelerator.prepare(
        noise, opt, model, dataloaders
    )
    
    # freeze model
    for name, param in model.named_parameters():
        param.requires_grad = False
    progress_bar = tqdm(range(num_epochs * len(dataloaders['harmful'])), desc="Training")
    step = 0
    total_loss = []
    for epoch in range(num_epochs):
        for harmful_batch, harmless_batch in zip(dataloaders['harmful'], dataloaders['harmless']):
            progress_bar.update(1)
            step += 1
            if harmful_batch['input_ids'].size(0) != initial_batch_size:
                continue
            with accelerator.accumulate(model):
                if use_embed:
                    inputs = noise(
                        harmful_batch['input_ids']
                    )
                else:
                    inputs = noise()
                mask = ~torch.eq(harmful_batch['input_ids'], harmless_batch['input_ids'])
                mask = mask.float().to(DEVICE)
                outputs = model(inputs_embeds=inputs.to(DEVICE))
                harmful_loss = masked_token_ce_loss(
                    outputs.logits,
                    harmful_batch['input_ids'],
                    mask
                )
                # i don't know if this sum is working i also think this factor dominates too much
                # TODO: replace with proper weight norm
                l2_norm = torch.mean(
                    torch.sum(inputs**2, dim=[1, 2]) # was [1, 2] could go back? seems to maximize harm on the second cycle morish
                )
                # l2_norm = torch.norm(
                #      torch.sum(inputs, dim=[1, 2]), p=2
                # )
                if minimize == True:
                    loss = harmful_loss + regularization_term * l2_norm
                else:
                    loss = -harmful_loss + regularization_term * l2_norm
                accelerator.backward(loss)
                logger.info("Loss: {}, harmful_loss {}".format(np.mean(total_loss), harmful_loss))
                opt.step()
                opt.zero_grad()
                total_loss.append(loss.cpu().detach().numpy())
    # unfreeze model
    # freeze model
    for name, param in model.named_parameters():
        param.requires_grad = True
    if use_embed == True:
        # freeze gradient
        noise.embedding.weight.requires_grad = False
        return noise
    else:
        return noise.noise.detach()

def train_model_simple(
    model_name,
    dataset_name,
    loss_fn_name,
    tokenizer,
    dataloaders,
    num_epochs=10,
    learning_rate=1e-3,
    adversarial_alpha=0.1,
    steps_to_eval=DEFAULT_STEPS_TO_EVAL,
    random_init=False,
    ntl_alpha=0.1,
    ntl_beta=0.1,
    ntl_num_layers=1,
    warmup_factor=0.1,
    scheduler="cosine",
    kernel_mul=2,
    kernel_num=5,
    batch_size=None,
    num_cycles=1,
    regularization_term=0.1,
    mask_path=None,
    sample=None,
    optimizer_name="adam",
    freeze=None
) -> Tuple[List[float], List[float]]:
    dtype = "auto"
    if 'meta-llama' in model_name or 'Qwen' in model_name:
        dtype = torch.bfloat16
    logger.info(f"Training model {model_name} on dataset {dataset_name} with loss function {loss_fn_name}")
    logger.info("Loading model...")
    model = None
    if loss_fn_name == "random":
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
        return model, {}
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)
    
    if freeze:
        print("Freezing")
        # lm_head lm_head_1 lm_head_4 ten_twenty 0_ten
        for name, param in model.named_parameters():
            if 'lm_head' in freeze and 'lm_head' in name:
                param.requires_grad = False
            if freeze == 'lm_head_1':
                if '31' in name:
                    param.requires_grad = False
            if freeze == 'lm_head_4':
                if any([ x in name for x in ["31", "30", "29", "28"]]):
                    param.requires_grad = False
            if freeze == "ten_twenty":
                if any([ x in name for x in ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]]):
                    param.requires_grad = False
            if freeze == "0_ten":
                if any([ x in name for x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]):
                    param.requires_grad = False
            if freeze == '20_31':
                if any([ x in name for x in ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]]):
                    param.requires_grad = False

    domain_classifier = None
    if loss_fn_name == 'ntl_aux':
        domain_classifier = classifier(model.config.hidden_size).to(accelerator.device)
        domain_classifier.train()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # right now for memory
    # TODO(dom): add weight decay and gradient clipping
    if loss_fn_name == 'ntl_aux':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': domain_classifier.parameters(), 'lr': 1e-3}
        ])
    else:
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adafactor':
            optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    if dataset_name in ['decoding_trust', 'beavertails', 'bthexphi']:
        num_training_steps = num_epochs * len(dataloaders['harmful'])
    else:
        num_training_steps = num_epochs * len(dataloaders['train'])
    # ToDo(dom): This scheduler might have an impact on immunization
    # should test!
    if optimizer_name == 'adam':
        lr_scheduler = get_scheduler(
            name=scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_training_steps * warmup_factor,
            num_training_steps=num_training_steps,
        )
    elif optimizer_name == 'adafactor':
        lr_scheduler = AdafactorSchedule(optimizer)
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    # prepare dataloaders dict
    for k, v in dataloaders.items():
        dataloaders[k] = accelerator.prepare(v)

    logger.info("Beginning training loop")
    # simple training loop
    
    if dataset_name in ['decoding_trust', 'beavertails', 'bthexphi']:
        if loss_fn_name == "ssd":
            model = ssd_tuning(
                model,
                dataloaders
            )
            return model, {}
        noise_vector = None
        if "explode" in loss_fn_name:
            for name, param in model.named_parameters():
                if 'lm_head' in name:
                    print(param)
                    param.data = torch.nn.Parameter(param.data * 2**61 - 1).to(DEVICE)
                    param.data += torch.randn(param.data.size()).to(DEVICE) * 0.01
            return model, {}
        if "security_vectors" in loss_fn_name:
            from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
            # train lora adapter with forzen model
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            peft_model = get_peft_model(model, peft_config)
            print(peft_model.get_nb_trainable_parameters())
            peft_optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-3)
            epochs = 1
            if dataset_name == 'decoding_trust':
                epochs = 4
            peft_model, _ = attack_or_immunization_loop(
                    dataset_name, "min_harmful_loss_no_eval", tokenizer, dataloaders,
                    epochs, adversarial_alpha, peft_model, peft_optimizer, 
                    steps_to_eval,
                    ntl_alpha,
                    ntl_beta,
                    ntl_num_layers,
                    domain_classifier=domain_classifier,
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                    noise_vector=noise_vector,
                    batch_size=batch_size,
                    mask_path=mask_path
                )
            
            if "1000_attack" in loss_fn_name:

                dataloaders['harmful'] = torch.utils.data.DataLoader(torch.utils.data.Subset(
                    dataloaders['harmful'].dataset, range(1000)
                ), batch_size=batch_size)
                dataloaders['harmless'] = torch.utils.data.DataLoader(torch.utils.data.Subset(
                    dataloaders['harmless'].dataset, range(1000)
                ), batch_size=batch_size)

            # run min_harmful_loss
            print(peft_model.get_nb_trainable_parameters())
            for name, param in peft_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(peft_model.get_nb_trainable_parameters())
            peft_model, data = attack_or_immunization_loop(
                    dataset_name, "min_harmful_loss", tokenizer, dataloaders,
                    1, adversarial_alpha, peft_model, optimizer, 
                    steps_to_eval,
                    ntl_alpha,
                    ntl_beta,
                    ntl_num_layers,
                    domain_classifier=domain_classifier,
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                    noise_vector=noise_vector,
                    batch_size=batch_size,
                    mask_path=mask_path
                )
            
            return peft_model, data 

        if "noise_loss_learned" in loss_fn_name:
            impair_learn_rate = 1e-3
            repair_learn_rate = 1e-3
            for i in range(num_cycles):
                logger.info(f"Noise cycle {i}")
                use_embed = False
                if "embed" in loss_fn_name:
                    use_embed = True
                minimize = False
                if "minimize" in loss_fn_name:
                    minimize = True
                noise_vector = learn_noise_vector(
                    model, dataloaders, tokenizer, num_epochs=1,
                    use_embed=use_embed, regularization_term=regularization_term,
                    minimize=minimize
                )
                # perform attack
                
                model, eval_data = attack_or_immunization_loop(
                    dataset_name, loss_fn_name, tokenizer, dataloaders,
                    num_epochs, adversarial_alpha, model, optimizer, 
                    steps_to_eval,
                    ntl_alpha,
                    ntl_beta,
                    ntl_num_layers,
                    domain_classifier=domain_classifier,
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                    noise_vector=noise_vector,
                    batch_size=batch_size,
                    mask_path=mask_path
                )
                #     perform repair
                model, _ = attack_or_immunization_loop(
                    dataset_name, "min_harmless_loss", tokenizer, dataloaders,
                    1, adversarial_alpha, model, optimizer, 
                    steps_to_eval,
                    ntl_alpha,
                    ntl_beta,
                    ntl_num_layers,
                    domain_classifier=domain_classifier,
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                    noise_vector=noise_vector,
                    batch_size=batch_size,
                    mask_path=mask_path
                )
        else:
            model, eval_data =  attack_or_immunization_loop(
                dataset_name, loss_fn_name, tokenizer, dataloaders,
                num_epochs, adversarial_alpha, model, optimizer, 
                steps_to_eval,
                ntl_alpha,
                ntl_beta,
                ntl_num_layers,
                domain_classifier=domain_classifier,
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                noise_vector=noise_vector,
                batch_size=batch_size,
                mask_path=mask_path,
                sample=sample,
                optimizer_name=optimizer_name
            )
        return model, eval_data

        

    return stability_train_loop(dataset_name, loss_fn_name, tokenizer, dataloaders, num_epochs, model, optimizer, lr_scheduler, steps_to_eval)


def attack_or_immunization_loop(
    dataset_name, loss_fn_name, tokenizer, dataloaders,
    num_epochs, adversarial_alpha, model, optimizer, 
    steps_to_eval,
    ntl_alpha=0.1,
    ntl_beta=0.1,
    ntl_num_layers=1,
    domain_classifier=None,
    kernel_mul=2,
    kernel_num=5,
    noise_vector=None,
    batch_size=None,
    mask_path=None,
    sample=None,
    optimizer_name='adam'
):
    mask = None
    if mask_path:
        logger.info('Loading mask for model')
        mask = torch.load(mask_path)
    warmup_factor = 0.1
    scheduler = 'cosine'
    num_training_steps = num_epochs * len(dataloaders['harmful'])
    if optimizer_name == 'adam' or not optimizer_name:
        lr_scheduler = get_scheduler(
            name=scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_training_steps * warmup_factor,
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = AdafactorSchedule(optimizer)
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    mmd_loss = MMD_loss(
        kernel_mul=kernel_mul,
        kernel_num=kernel_num
    )
    eval_datas = []
    # DEBUG just for finding immunization model no need to eval
    # if loss_fn_name in ["min_harmful_loss", "vaccine"]: 
    #     eval_data = attack_or_immunization_eval(
    #         model, tokenizer, dataloaders, 0, 0, dataset_name,
    #         sample=sample
    #     )
    #     eval_datas.append(eval_data)
    #     if os.environ.get('WANDB_API_KEY'):
    #         wandb.log(eval_data)
    progress_bar = tqdm(range(num_epochs * len(dataloaders['harmful'])), desc="Training")
    # we should evaluate in four chunks since its expensive timewise to evaluate
    #steps_to_eval = num_epochs * len(dataloaders['harmful']) // 4
    # if os.environ.get('WANDB_API_KEY'):
    #     wandb.watch(model, log_freq=100, log="all")
    step = 0
    for epoch in range(num_epochs):
        logger.info(f"EPOCH {epoch} of {num_epochs}")
        for harmful_loss, harmless_batch in zip(dataloaders['harmful'], dataloaders['harmless']):
            progress_bar.update(1)
            step += 1
            model.train()
            # if loss_fn_name != "min_harmful_loss":
            #     layers = [
            #         name for name in
            #         model.named_parameters()
            #     ]
            #     # select random number of layers based len(layers)
            #     num_layers_to_select = random.randint(0, len(layers) // 10)
            #     layers_to_freeze = random.sample(layers, num_layers_to_select)
            #     # last n layers
            #     #layers_to_freeze = layers[-num_layers_to_select:]
            #     for name, param in model.named_parameters():
            #         if name in layers_to_freeze:
            #             param.requires_grad = False
            with accelerator.accumulate(model):

                loss = compute_loss(
                    loss_fn_name, adversarial_alpha, model, ntl_alpha, ntl_beta, 
                    ntl_num_layers, domain_classifier, kernel_mul, kernel_num, mmd_loss,
                    harmful_loss, harmless_batch, tokenizer, step, num_epochs * len(dataloaders['harmful']),
                    noise_vector=noise_vector, batch_size=batch_size
                )
                if loss:
                    accelerator.backward(loss)
                logger.info(f"Loss: {loss}")
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name].to(param.grad.device)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # if loss_fn_name != "min_harmful_loss":
            #     # unfreeze
            #     for name, param in model.named_parameters():
            #         if name in layers_to_freeze:
            #             param.requires_grad = True
            if os.environ.get('WANDB_API_KEY'):
                wandb.log({
                    "train_loss": loss
                })
            # if loss_fn_name in ["min_harmful_loss", "vaccine"]:
            #     if step != 0 and step % steps_to_eval == 0:
            #         eval_data = attack_or_immunization_eval(
            #             model, tokenizer, dataloaders, epoch, step, dataset_name,
            #             sample=sample
            #         )
            #         eval_datas.append(eval_data)
            #         if os.environ.get('WANDB_API_KEY'):
            #             wandb.log(eval_data)
                    # if eval_data["downstream_performance"]["harmfulness_scores"]["score"] > 0.7 and dataset_name == "beavertails":
                    #     return model, eval_datas
                    # if eval_data["downstream_performance"]["harmfulness_scores"]["score"] > 0.7 and dataset_name == "decoding_trust":
                    #     return model, eval_datas
    if loss_fn_name in ["min_harmful_loss", "vaccine"]:
        eval_data = attack_or_immunization_eval(
            model, tokenizer, dataloaders, epoch, step, dataset_name
        )
        eval_datas.append(eval_data)
    return model, eval_datas

def compute_loss(
        loss_fn_name, adversarial_alpha, model, ntl_alpha, ntl_beta,
        ntl_num_layers, domain_classifier, kernel_mul, kernel_num, mmd_loss,
        harmful_loss, harmless_batch, tokenizer, step, total_steps,
        noise_vector=None, batch_size=None
):
    if "min_harmful_loss" in loss_fn_name:
        outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'])
        loss = outputs.loss
    elif loss_fn_name == "min_harmless_l2_explosion":
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'], output_hidden_states=True)
        l2_norm = 0
        for hidden in harmless_outputs.hidden_states:
            l2_norm += torch.mean(
                torch.sum(hidden**2, dim=[1, 2]) # was [1, 2] could go back? seems to maximize harm on the second cycle morish
            )
        # can try to get some other bits like norm, input embeddings, and output embeddings
        l2_norm = l2_norm / len(harmless_outputs.hidden_states)

        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "norm": l2_norm
        })
        logger.info(f"Loss parts: harmless_ouputs_loss: {harmless_outputs.loss}, norm: {l2_norm}")
        loss = harmless_outputs.loss - 0.01 * torch.log(l2_norm)
    elif loss_fn_name == "max_harmful_loss":
        outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'])
        loss = -outputs.loss
    elif loss_fn_name == 'min_harmless_loss':
        outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        loss = outputs.loss
    elif loss_fn_name == 'adversarial_loss':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'])
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss
        })
        logger.info(f"Loss parts: harmless_ouputs_loss: {harmless_outputs_loss} {harmless_outputs.loss}, harmful_outputs_loss: {harmful_outputs_loss} {harmful_outputs.loss}")
        loss = harmless_outputs_loss - adversarial_alpha * harmful_outputs_loss
    elif loss_fn_name == 'random_label_loss':
        random_labels = construct_random_labels(
            harmful_loss,
            tokenizer
        ).to(DEVICE)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=random_labels)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss
        })
        logger.info(f"Loss parts: harmless_ouputs_loss: {harmless_outputs_loss} {harmless_outputs.loss}, harmful_outputs_loss: {harmful_outputs_loss} {harmful_outputs.loss}")
        loss = harmless_outputs_loss + harmful_outputs_loss
    elif 'noise_loss_learned_contrastive' in loss_fn_name:
        if harmful_loss['input_ids'].size(0) != batch_size:
            return None
        use_embed = False
        if "embed" in loss_fn_name:
            use_embed = True
        if use_embed:
            noise_outputs = model(inputs_embeds=noise_vector(harmful_loss['input_ids']), labels=harmful_loss['input_ids'])
        else:
            noise_outputs = model(inputs_embeds=noise_vector, labels=harmful_loss['input_ids'])
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['input_ids'], labels=harmful_loss['input_ids'])
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        noise_outputs_loss = masked_token_ce_loss(
            noise_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss,
            "noise_loss": noise_outputs_loss
        })
        logger.info(f"Noise Loss parts: noise_outputs_loss {noise_outputs_loss} harmless_ouputs_loss: {harmless_outputs_loss}, harmful_outputs_loss: {harmful_outputs_loss} ")
        loss = harmless_outputs_loss + noise_outputs_loss - 0.1 * harmful_outputs_loss
    elif loss_fn_name == 'noise_loss':
        random_labels = construct_random_labels(
            harmful_loss,
            tokenizer
        ).to(DEVICE)
        harmful_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'])
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss
        })
        logger.info(f"Noise Loss parts: harmless_ouputs_loss: {harmless_outputs_loss}, harmful_outputs_loss: {harmful_outputs_loss} ")
        loss = harmless_outputs_loss + harmful_outputs_loss
    elif 'noise_loss_learned' in loss_fn_name:
        if harmful_loss['input_ids'].size(0) != batch_size:
            return None
        use_embed = False
        if "embed" in loss_fn_name:
            use_embed = True
        if use_embed:
            harmful_outputs = model(inputs_embeds=noise_vector(harmful_loss['input_ids']), labels=harmful_loss['input_ids'])
        else:
            harmful_outputs = model(inputs_embeds=noise_vector, labels=harmful_loss['input_ids'])
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss
        })
        logger.info(f"Noise Loss parts: harmless_ouputs_loss: {harmless_outputs_loss}, harmful_outputs_loss: {harmful_outputs_loss} ")
        loss = harmless_outputs_loss + harmful_outputs_loss
    
    elif loss_fn_name == 'mmd':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        mmd = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        logger.info(f"MMD: {mmd}")
        wandb.log({
            "MMD": mmd
        })
        loss = -mmd
    
    elif loss_fn_name == 'ntl_ce':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True)
                
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        mmd = ntl_beta * mmd_org
        l_aux = harmful_outputs.loss * ntl_alpha
        distances = decorrelation_loss(
            hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )


        logger.info(f"Loss parts: harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} mmd: {mmd_org} l_aux: {l_aux}")
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss,
            'mmd': mmd_org,
            'distances': distances
        })
        loss = harmless_outputs.loss - l_aux * mmd
    elif loss_fn_name == 'ce_observe':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True)
                
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        mmd = ntl_beta * mmd_org
        l_aux = harmful_outputs.loss * ntl_alpha
        distances = decorrelation_loss(
            hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )


        logger.info(f"Loss parts: 'distances': {distances} harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} mmd: {mmd_org} l_aux: {l_aux}")
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss,
            'mmd': mmd_org,
            'distances': distances
        })
        loss = harmless_outputs.loss - l_aux
    elif loss_fn_name == 'ntl_layer_random':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )          
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # choose 6 random layer numbers and grab those
        # TODO: try contigious random selection
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]
        mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * mask
        hiddens_harmful = hiddens_harmful * mask
                    
        mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        mmd = ntl_beta * mmd_org
        l_aux = harmful_outputs_loss * ntl_alpha
        # distances = decorrelation_loss(
        #     hiddens_harmful.view(hiddens_harmful.size(0), -1)
        # )
        logger.info(f"Loss parts: harmless_ce: {harmless_outputs_loss} harmful ce: {harmful_outputs_loss} mmd: {mmd_org} l_aux: {l_aux}")
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss,
            "mmd": mmd_org,
            # "distances": distances
        })
        loss = harmless_outputs_loss - l_aux - mmd
    elif loss_fn_name == 'ntl_layer_random_noise':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True)
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        with torch.no_grad():
            random_labels = construct_random_labels(
                harmful_loss,
                tokenizer
            ).to(DEVICE)
            noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        num_layers = hiddens_harmful.size(1)
        # TODO: try masking
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]
        hiddens_noise = hiddens_noise[:, layer_idxs, :, :]
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])

        # Convert the mask to float
        mask = mask.float()

        mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * mask
        hiddens_harmful = hiddens_harmful * mask
        hiddens_noise = hiddens_noise * mask
        
        # mmd is a little more stable less collapse
        noise_mmd_org = mmd_loss(hiddens_noise.view(hiddens_noise.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # noise_mmd_org = torch.log(criterion_KL(
        #     F.log_softmax(hiddens_harmful, dim=-1),
        #     F.softmax(hiddens_noise, dim=-1)
        # ))
        mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        mmd =  mmd_org # ntl_beta *
        l_aux = harmful_outputs_loss # * ntl_alpha
        
        logger.info(f"Loss parts: noise_mmd_org: {noise_mmd_org} harmless_ce: {harmless_outputs_loss} harmful ce: {harmful_outputs_loss} mmd: {mmd_org} l_aux: {l_aux}")
        wandb.log({
            "harmless_loss": harmless_outputs_loss,
            "harmful_loss": harmful_outputs_loss,
            "mmd": mmd_org,
            "noise_mmd_org": noise_mmd_org
        })
        # log scale laux?
        loss = harmless_outputs_loss + noise_mmd_org * ntl_beta - torch.log(l_aux) - mmd 
    elif loss_fn_name == 'curvature':
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids']) # output_hidden_states=True
        # harmful_to_harmless_outputs = model(harmful_loss['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        # random_labels = construct_random_labels(
        #     harmful_loss,
        #     tokenizer
        # ).to(DEVICE)
        # harmful_outputs_noise_loss = cross_entropy_loss(
        #     harmful_outputs.logits,
        #     random_labels
        # )
        # noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # num_layers = hiddens_harmful.size(1)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful [:, layer_idxs, :, :]
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]
        # hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)[:, layer_idxs, :, :]
                    
        # mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # mmd = ntl_beta * mmd_org
        # noise_mmd_org = mmd_loss(hiddens_noise.view(hiddens_noise.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # noise_mmd = ntl_beta * mmd_org
        # l_aux = harmful_outputs.loss * ntl_alpha
        # distances = decorrelation_loss(
        #     hiddens_harmful.view(hiddens_harmful.size(0), -1)
        # )
        # kl_div = criterion_KL(
        #     F.log_softmax(hiddens_harmful, dim=-1),
        #     F.softmax(torch.rand_like(hiddens_harmful), dim=-1)
        # )
        increase_param = (
            (total_steps - step) / total_steps
        ) * 1.5
        curvature, grad_norm = regularizer(model, harmful_loss['input_ids'], harmful_loss['input_ids'], h=increase_param)
        #harmless_curvature, harmless_grad_norm = regularizer(model, harmless_batch['input_ids'], harmless_batch['input_ids'], h=increase_param)
        logger.info(f" curvature {curvature.item()}, grad_norm {grad_norm} harmless_outputs.loss {harmless_outputs.loss} harmful_outputs: {harmful_outputs.loss} ") # harmful_to_harmless_loss: {harmful_to_harmless_outputs.loss} distances: {distances} noise loss: {harmful_outputs_noise_loss} noise mmd: {noise_mmd_org} harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} mmd: {mmd_org} l_aux: {l_aux}
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss,
            "curvature": curvature,
            "grad norm": grad_norm
        })
        # loss = harmless_outputs.loss + harmful_to_harmless_outputs.loss + 0.01 * harmful_outputs_noise_loss + noise_mmd - l_aux - mmd
        loss = harmless_outputs.loss - curvature * 0.001 - harmful_outputs.loss * 0.5
    elif loss_fn_name == 'minimality':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True) # output_hidden_states=True
        # harmful_to_harmless_outputs = model(harmful_loss['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'])
        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # num_layers = hiddens_harmful.size(1)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful [:, layer_idxs, :, :]
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)
                    
        # mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # mmd = ntl_beta * mmd_org
        # noise_mmd_org = mmd_loss(hiddens_noise.view(hiddens_noise.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # noise_mmd = ntl_beta * mmd_org
        # l_aux = harmful_outputs.loss * ntl_alpha
        # distances = decorrelation_loss(
        #     hiddens_harmful.view(hiddens_harmful.size(0), -1)
        # )
        # mask = torch.eq(hiddens_harmful, hiddens_harmless)
        # # Invert the mask
        # invmask = ~mask
        # # Convert the mask to float
        # invmask = invmask.float()
        # # Apply the mask to zero out common values
        # hiddens_harmful = hiddens_harmful * mask
        gaussian = torch.rand_like(hiddens_harmful) # * mask
        temperature = 1
        # noise_mmd_org = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # noise_mmd = ntl_beta * noise_mmd_org
        # # I should apply a representation mask on the harmful bits
        kl_div = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        ) 
        # # curvature, grad_norm = regularizer(model, harmful_loss['input_ids'], harmful_loss['input_ids'])
        # logger.info(f" Loss parts: noise mmd: {noise_mmd_org}  harmful_outputs: {harmful_outputs.loss} ") # harmful_to_harmless_loss: {harmful_to_harmless_outputs.loss} distances: {distances} noise loss: {harmful_outputs_noise_loss} noise mmd: {noise_mmd_org} harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} mmd: {mmd_org} l_aux: {l_aux}
        # wandb.log({
        #    # "harmless_loss": harmless_outputs.loss,
        #     "harmful_loss": harmful_outputs.loss,
        #     "noise mmd": noise_mmd_org
        #     # "kl div": kl_div
        # })
        # loss = harmless_outputs.loss + harmful_to_harmless_outputs.loss + 0.01 * harmful_outputs_noise_loss + noise_mmd - l_aux - mmd
        #loss = harmless_outputs.loss  + noise_mmd * 0.001 - harmful_outputs.loss * 0.5# - curvature * 0.01
        # Initialize the weight norm
        # wn = 0.0

        # # Iterate over the model parameters
        # for param in model.parameters():
        #     # Compute the Frobenius norm of the parameter tensor and add it to the weight norm
        #     wn += torch.norm(param.to(DEVICE))

        # print(f'Weight norm: {wn}')
        logger.info(f"Loss parts: harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} kl_div: {kl_div}")
        loss = harmless_outputs.loss + domain_loss - mmd
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss,
            "kl_div": kl_div
        })
        # if step % 100 == 0:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist())[0])
        loss = torch.log(kl_div)
    elif loss_fn_name == 'ntl_aux':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True)
                
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)[:, -ntl_num_layers:, :, :]
        mmd_org = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        mmd = ntl_beta * mmd_org
        l_aux = harmful_outputs.loss * ntl_alpha
                    
        batch_size = harmful_loss['input_ids'].size(0)
        zeros = torch.tensor([0 for _ in range(batch_size)]).cuda()
        ones = torch.tensor([1 for _ in range(batch_size)]).cuda()
        l_domain_1 = ce_loss(domain_classifier(hiddens_harmful[-1, :, -1, :].squeeze()).view(-1, 2), zeros) # [-1, :, -1, :] .mean(dim=0).mean(dim=1)
        l_domain_2 = ce_loss(domain_classifier(hiddens_harmless[-1, :, -1, :].squeeze()).view(-1, 2), ones) # [-1, :, -1, :] .mean(dim=0).mean(dim=1)
        domain_loss = (l_domain_1 + l_domain_2) * 0.5

        logger.info(f"Loss parts: harmless_cd: {harmless_outputs.loss} harmful ce: {harmful_outputs.loss} mmd: {mmd_org} l_aux: {l_aux} domain loss: {domain_loss}")
        loss = harmless_outputs.loss + domain_loss - mmd
        wandb.log({
            "harmless_loss": harmless_outputs.loss,
            "harmful_loss": harmful_outputs.loss,
            "domain loss": domain_loss,
            "mmd": mmd
        })
    elif loss_fn_name == 'lens-loss':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'], labels=harmless_batch['input_ids'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'], labels=harmful_loss['input_ids'],output_hidden_states=True)

        output_embeddings = model.get_output_embeddings()
        output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        norm.requires_grad = False

        harmless_losses = harmless_outputs.loss
        logger.info(f"Loss parts: harmless losses {harmless_losses}")
        for i, h in enumerate(harmless_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = cross_entropy_loss(
                out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE)
            )
            harmless_losses += loss
        harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs.loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = cross_entropy_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE)
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1
        
        loss = harmless_losses - 2 * torch.log(harmful_losses)
        logger.info(f"Loss parts: harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
        })
    
    elif loss_fn_name == 'adv-mmd':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        mmd = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))

        loss = harmless_losses - 2 * torch.log(harmful_losses) - mmd
        logger.info(f"Loss parts: mmd: {mmd}  harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "mmd": mmd
        })
    elif loss_fn_name == 'lens-loss-minimality-reverse':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        # std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        # y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        # std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        # std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        if ntl_num_layers < num_layers:
            layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        else:
            layer_idxs = num_layers
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        )
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses - 2 * torch.log(harmful_losses) - 4 * torch.log(noise_mmd)
        
        logger.info(f"Loss parts: snoise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
   
        })
    elif loss_fn_name == 'lens-loss-minimality-reverse-mmd':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        # std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        # y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        # std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        # std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        if ntl_num_layers < num_layers:
            layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        else:
            layer_idxs = num_layers
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses  - 2 * torch.log(harmful_losses) - 4 * noise_mmd
        
        logger.info(f"Loss parts:  snoise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
   
        })
    elif loss_fn_name == 'lens-loss-minimality-mmdreal':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.rand_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + noise_mmd * 1 + std_loss - 2 * torch.log(harmful_losses) - mmd
        # print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        # print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'lens-loss-minimality-mmd-no-adv':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.rand_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        )
        mmd = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + torch.log(noise_mmd) * 1 + std_loss - mmd
        # print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        # print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Loss parts: mmd: {mmd} std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss,
            "mmd": mmd
        })
    elif loss_fn_name == 'lens-loss-mmd':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        

        temperature = 1
        mmd = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses - torch.log(harmful_losses) - mmd
        # print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        # print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Loss parts: mmd: {mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "mmd": mmd
        })
    elif loss_fn_name == 'lens-loss-minimality-l2':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])

        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = torch.cdist(
            hiddens_harmful, gaussian
        ).mean()
       
        loss = harmless_losses + noise_mmd + std_loss - 2 * torch.log(harmful_losses)
        logger.info(f"Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'rate':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        # std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        # y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        # std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        # std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        if ntl_num_layers < num_layers:
            layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        else:
            layer_idxs = num_layers
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask

        mean = torch.mean(hiddens_harmful.view(hiddens_harmful.size(0), -1), dim=-1).float()
        
        covariance = torch.cov(hiddens_harmful.view(hiddens_harmful.size(0), -1)).float()
        from torch.distributions import MultivariateNormal
        # # Create a multivariate normal distribution
        try:
            dist = MultivariateNormal(mean, covariance)
            # # Sample from the distribution
            marginal = MultivariateNormal(torch.zeros(mean.shape).to(mean.device), torch.eye(covariance.size(0)).to(covariance.device))

            z_samps = dist.sample((16,))
            rate = (dist.log_prob(z_samps) - marginal.log_prob(z_samps)).mean(0)
            noise_mmd = rate
        except:
            noise_mmd = 0
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + noise_mmd - 2 * torch.log(harmful_losses)
        
        logger.info(f"Loss parts: snoise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
   
        })
    elif loss_fn_name == 'minimality-per-layer':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # # x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        # # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        # # std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        # # y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        # # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        # # std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        # # std_loss = std_loss_x + std_loss_y
        # num_layers = hiddens_harmful.size(1)
        # if ntl_num_layers < num_layers:
        #     layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # else:
        #     layer_idxs = num_layers
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # # std can prevent collapse: VICReg paper
        
        # #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        #hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # # hiddens_noise = hiddens_noise * mask
         
        # gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        noise_mmd = 0
        for hidden in harmful_outputs.hidden_states:
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size())
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens) * hiddens_mask
            # clip the dim 1 to the longest sequence that is non-zero for effeciency
            print(
                hiddens_mask.sum(0).max(),
                hiddens_mask.sum(1).max(),
                hiddens_mask.sum(-1).max()
            )
            print(hiddens.shape)
            
            hiddens = hiddens[:, :hiddens_mask.sum(1).max().long(), :]
            gaussian = gaussian[:, :hiddens_mask.sum(1).max().long(), :]

            # distance function
            noise_mmd += criterion_KL(
                F.log_softmax(hiddens.view(hiddens.size(0), -1), dim=-1),
                F.softmax(gaussian.view(gaussian.size(0), -1) , dim=-1)
            )
        noise_mmd /= len(harmful_outputs.hidden_states)
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + 4 * torch.log(noise_mmd) - 2 * torch.log(harmful_losses) 
        
        logger.info(f"Loss parts: snoise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
   
        })
    elif loss_fn_name == 'minimality-per-layer-reverse':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # # x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        # # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        # # std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        # # y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        # # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        # # std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        # # std_loss = std_loss_x + std_loss_y
        # num_layers = hiddens_harmful.size(1)
        # if ntl_num_layers < num_layers:
        #     layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # else:
        #     layer_idxs = num_layers
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # # std can prevent collapse: VICReg paper
        
        # #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # # hiddens_noise = hiddens_noise * mask
         
        # gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        noise_mmd = 0
        for hidden in harmful_outputs.hidden_states:
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size())
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens) * hiddens_mask
            # distance function
            noise_mmd += criterion_KL(
                F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1), dim=-1),
                F.softmax(gaussian.view(gaussian.size(0), -1) , dim=-1)
            )
        noise_mmd /= len(harmful_outputs.hidden_states)
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses - 4 * torch.log(noise_mmd) - 2 * torch.log(harmful_losses) 
        
        logger.info(f"Loss parts: snoise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
   
        })

    elif loss_fn_name == 'minimality-mmd':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = len(harmful_outputs.hidden_states)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'ascent_loss':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        
        norm = model.base_model.norm 
        
        
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / (len(harmful_outputs.hidden_states) + 1)

        loss = harmless_losses - ntl_alpha * torch.log(harmful_losses)
        
        
        logger.info(f"Loss parts:  harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        
    elif loss_fn_name == 'vaccine':
        loss = vaccine_training_step(
            model, harmful_loss, RHO=ARGS.vaccine_rho
        )
        logger.info(f"Loss {loss} ")
        wandb.log({
            "harmful losses": loss
        })
        loss = None
    elif loss_fn_name == 'minimality-mmd-without-lens':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        
        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-without-mask':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], labels=harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden
            gaussian = torch.randn_like(hiddens).to(hidden.device)
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = harmful_outputs.loss
        harmless_outputs = model(harmless_batch['input_ids'], labels=harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = harmless_outputs.loss
        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-without-mask-and-lens':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], labels=harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden
            gaussian = torch.randn_like(hiddens).to(hidden.device)
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = harmful_outputs.loss
        harmless_outputs = model(harmless_batch['input_ids'], labels=harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = harmless_outputs.loss
        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss

        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    
    elif loss_fn_name == 'minimality-mmd-reverse':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        loss = -(harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses))
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'repnoise-svd-topk':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            proj = torch.linalg.svdvals(hiddens.float())
            # select top k singular values
            proj = proj[:256]
            gaussian = torch.randn_like(proj).to(hidden.device)
            # distance function
            noise_mmd += mmd_loss(proj.view(proj.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = len(harmful_outputs.hidden_states)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'repremove':
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])

        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1
        loss = harmless_losses - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts: harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
        })
    elif loss_fn_name == 'minimality-mmd-no-ascent':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = len(harmful_outputs.hidden_states)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + ntl_beta * noise_mmd  #- ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-no-noise':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(len(model.base_model.layers)):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = len(harmful_outputs.hidden_states)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-l2':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(32):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            # noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
            noise_mmd += l2_distance(
                hiddens.view(hiddens.size(0), -1),
                gaussian.view(gaussian.size(0), -1)
            ).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = len(harmful_outputs.hidden_states)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + 4 * torch.log(noise_mmd) - 2 * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-ab':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for i in range(31):
            harmful_activations.append(activations[f'model.layers.{i}.mlp'])
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        num_layers = len(harmful_outputs.hidden_states)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + ntl_beta * noise_mmd - ntl_alpha * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-all':

        activations = register_activation_hook(model)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        harmful_activations = []
        for activation in activations.values():
            harmful_activations.append(activation)
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        noise_mmd = 0
        for i, hidden in enumerate(harmful_activations):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
        noise_mmd /= len(harmful_activations) # len(layer_idxs)
        print(noise_mmd)
        
        
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        output_embeddings = model.get_output_embeddings()
        # output_embeddings.requires_grad = False
        norm = model.base_model.norm 
        
        # norm.requires_grad = False
        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses /= len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        for i, h in enumerate(harmful_outputs.hidden_states):
            out = output_embeddings(norm(h))
            loss = masked_token_ce_loss(
                out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
                mask
            )
            harmful_losses += loss
        harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        num_layers = len(harmful_outputs.hidden_states)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_nameqq
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + 4 * noise_mmd - 2 * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-l2':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        num_layers = len(harmful_outputs.hidden_states)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        noise_mmd = 0
        for i, hidden in enumerate(harmful_outputs.hidden_states):
            # if i not in layer_idxs:
            #     continue
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size())
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens) * hiddens_mask
            # distance function
            mmd = mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1))
            l2 = l2_distance(
                hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)
            )
            print(l2)
            print(mmd)
            noise_mmd += torch.log(l2) + mmd
        noise_mmd /= len(harmful_outputs.hidden_states)
        loss = harmless_losses + 4 * noise_mmd - 2 * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'minimality-mmd-reverse':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        # logger.info(f"Loss parts: harmless losses {harmless_losses}")
        # for i, h in enumerate(harmless_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmless_batch['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmless_losses += loss
        # harmless_losses = harmless_losses / len(harmless_outputs.hidden_states) + 1
        harmful_losses = harmful_outputs_loss
        # logger.info(f"Loss parts: harmful losses {harmful_losses} ")
        # for i, h in enumerate(harmful_outputs.hidden_states):
        #     out = output_embeddings(norm(h))
        #     loss = masked_token_ce_loss(
        #         out.to(DEVICE), harmful_loss['input_ids'].to(DEVICE),
        #         mask
        #     )
        #     harmful_losses += loss
        # harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

        # with torch.no_grad():
        #     random_labels = construct_random_labels(
        #         harmful_loss,
        #         tokenizer
        #     ).to(DEVICE)
        #     # harmful_outputs_noise_loss = cross_entropy_loss(
        #     #     harmful_outputs.logits,
        #     #     random_labels
        #     # )
        #     noise_outputs = model(random_labels, attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        # hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        # hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        # num_layers = hiddens_harmful.size(1)
        # layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        # hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        # hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        # hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # # apply the mask
        # hiddens_harmless = hiddens_harmless * hiddens_mask
        # hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
        noise_mmd = 0
        for hidden in harmful_outputs.hidden_states:
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size())
            # expand(torch.cuda.FloatTensor{[8, 1, 256, 1]}, size=[8, 256, 4096])
            hiddens = hidden * hiddens_mask
            gaussian = torch.randn_like(hiddens) * hiddens_mask
            # distance function
            noise_mmd += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1))
        noise_mmd /= len(harmful_outputs.hidden_states)
        # temperature = 1
        # # noise_mmd = criterion_KL(
        # #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        # #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # # )
        # noise_mmd = mmd_loss(hiddens_harmful.view(hiddens_harmful.size(0), -1), gaussian.view(gaussian.size(0), -1))
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses - 4 * noise_mmd - 2 * torch.log(harmful_losses)
        
        # loss = noise_mmd
        logger.info(f"Loss parts:  noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
        })
    elif loss_fn_name == 'lens-loss-minimality':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])

        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        )
       
        loss = harmless_losses + torch.log(noise_mmd) - 2 * torch.log(harmful_losses)
        logger.info(f"Loss parts: noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd
        })
    elif loss_fn_name == 'lens-loss-minimality-noise-min':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        noise_vector = torch.randn_like(model.get_input_embeddings().weight[
            harmful_loss['input_ids']
        ])
        noise_to_harm_outputs = model(inputs_embeds=noise_vector, labels=harmful_loss['input_ids'])
        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask

        temperature = 1
        noise_mmd = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        )
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + noise_to_harm_outputs.loss + 1 * torch.log(noise_mmd) + std_loss - 2 * torch.log(harmful_losses)
        # print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        # print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'lens-loss-minimality-estimated':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        # output_embeddings = model.get_output_embeddings()
        # # output_embeddings.requires_grad = False
        # norm = model.base_model.norm 
        # norm.requires_grad = False
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )

        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        # std can prevent collapse: VICReg paper
        
        #hiddens_noise = torch.stack(noise_outputs.hidden_states, dim=1)

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask
        # hiddens_noise = hiddens_noise * mask
         
        gaussian = torch.randn_like(
            hiddens_harmful
        ) * hiddens_mask

        temperature = 1
        noise_mmd = criterion_KL(
            F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
            F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        )
        
        # harmless_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmless.view(hiddens_harmless.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1) / temperature, dim=-1)
        # ) * 0.01
        #noise_mmd = ntl_beta * noise_mmd_org
        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         hiddens_harmful,
        #         hiddens_harmless,
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + ntl_alpha * torch.log(noise_mmd) + std_loss - ntl_beta * torch.log(harmful_losses)
        # print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        # print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'lens-loss-minimality-noise':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # std can prevent collapse: VICReg paper
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask


        temperature = 1
        # noise_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # )
        # noise_mmd = torch.log(noise_mmd)
        noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))

        

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         harmful_outputs.hidden_states[-1],
        #         harmless_outputs.hidden_states[-1],
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + noise_mmd + std_loss - 2 * torch.log(harmful_losses)
        print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Step: {step} Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'lens-loss-minimality-noise-mmd':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)

        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        
        harmless_losses = harmless_outputs_loss
        
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # std can prevent collapse: VICReg paper
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask


        temperature = 1
        # noise_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # )
        # noise_mmd = torch.log(noise_mmd)
        noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         harmful_outputs.hidden_states[-1],
        #         harmless_outputs.hidden_states[-1],
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        mmd = mmd_loss(hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + torch.log(noise_mmd) + std_loss - mmd - 2 * torch.log(harmful_losses)
        print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0])
        logger.info(f"Step: {step} Loss parts: mmd: {mmd} std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise mmd {noise_mmd} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_mmd": noise_mmd,
            "std_loss": std_loss,
            "mmd": mmd
        })
    elif loss_fn_name == 'l2-noise-distance':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        
        harmless_losses = harmless_outputs_loss
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # std can prevent collapse: VICReg paper
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask


        temperature = 1
        # noise_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # )
        # noise_mmd = torch.log(noise_mmd)
        # noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        noise_l2 = l2_distance(
            gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )
        harmful_harmless_l2 = l2_distance(
            hiddens_harmless.view(hiddens_harmless.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         harmful_outputs.hidden_states[-1],
        #         harmless_outputs.hidden_states[-1],
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + torch.log(noise_l2) + std_loss - torch.log(harmful_harmless_l2) - 2 * torch.log(harmful_losses)
        print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        logger.info(f"Step: {step} Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise l2 {noise_l2} harmful_harmless_l2 {harmful_harmless_l2} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_l2": noise_l2,
            "harmful_harmless_l2": harmful_harmless_l2,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'l2-noise':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        
        harmless_losses = harmless_outputs_loss
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # std can prevent collapse: VICReg paper
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask


        temperature = 1
        # noise_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # )
        # noise_mmd = torch.log(noise_mmd)
        # noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        noise_l2 = l2_distance(
            gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         harmful_outputs.hidden_states[-1],
        #         harmless_outputs.hidden_states[-1],
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = harmless_losses + torch.log(noise_l2) + std_loss - 2 * torch.log(harmful_losses)
        print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        logger.info(f"Step: {step} Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} noise l2 {noise_l2} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "noise_l2": noise_l2,
            "std_loss": std_loss
        })
    elif loss_fn_name == 'scatter':
        harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
        harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'],output_hidden_states=True)
        
        mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
        # Convert the mask to float
        mask = mask.float().to(DEVICE)
        harmful_outputs_loss = masked_token_ce_loss(
            harmful_outputs.logits,
            harmful_loss['input_ids'],
            mask
        )
        harmless_outputs_loss = masked_token_ce_loss(
            harmless_outputs.logits,
            harmless_batch['input_ids'],
            mask
        )
        
        harmless_losses = harmless_outputs_loss
        harmful_losses = harmful_outputs_loss
        
        hiddens_harmful = torch.stack(harmful_outputs.hidden_states, dim=1)
        hiddens_harmless = torch.stack(harmless_outputs.hidden_states, dim=1)
        # std can prevent collapse: VICReg paper
        x = hiddens_harmless - hiddens_harmless.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2
        y = hiddens_harmful- hiddens_harmful.mean(dim=0)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = std_loss_x + std_loss_y
        harm_cov = pairwise_covariance(hiddens_harmful.view(hiddens_harmful.size(0), -1))
        print('harm_cov', harm_cov)
        print('harmless_cov', pairwise_covariance(hiddens_harmless.view(hiddens_harmless.size(0), -1)))
        num_layers = hiddens_harmful.size(1)
        layer_idxs = random.sample(range(num_layers), ntl_num_layers)
        hiddens_harmful = hiddens_harmful[:, layer_idxs, :, :]
        hiddens_harmless = hiddens_harmless[:, layer_idxs, :, :]

        
        hiddens_mask = mask.unsqueeze(1).unsqueeze(-1).expand(hiddens_harmless.size())
        # apply the mask
        gaussian = torch.randn_like(hiddens_harmful) * hiddens_mask
        hiddens_harmless = hiddens_harmless * hiddens_mask
        hiddens_harmful = hiddens_harmful * hiddens_mask


        temperature = 1
        # noise_mmd = criterion_KL(
        #     F.log_softmax(hiddens_harmful.view(hiddens_harmful.size(0), -1)/ temperature, dim=-1),
        #     F.softmax(gaussian.view(gaussian.size(0), -1) / temperature, dim=-1)
        # )
        # noise_mmd = torch.log(noise_mmd)
        # noise_mmd = mmd_loss(gaussian.view(gaussian.size(0), -1), hiddens_harmful.view(hiddens_harmful.size(0), -1))
        pairwise_l2 = l2_pairwise(
            hiddens_harmful.view(hiddens_harmful.size(0), -1)
        )

        # if step % 100 == 0 or step == 1:
        #     save_pca(
        #         harmful_outputs.hidden_states[-1],
        #         harmless_outputs.hidden_states[-1],
        #         step=step,
        #         loss_fn_name=loss_fn_name
        #     )
        # problem is we want the harmful_losses to only apply to the n different tokens...
        loss = torch.abs(harm_cov) # - 2 * torch.log(harmful_losses)
        print(tokenizer.batch_decode(harmless_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        print(tokenizer.batch_decode(harmful_outputs.logits.argmax(-1).squeeze().tolist(), skip_special_tokens=True)[0].strip())
        logger.info(f"Step: {step} cov: {harm_cov} Loss parts: std_x {std_loss_x} std_y {std_loss_y} std_loss {std_loss} pairwise l2 {pairwise_l2} harmless losses {harmless_losses} harmful losses {harmful_losses} ")
        wandb.log({
            "harmless losses": harmless_losses,
            "harmful losses": harmful_losses,
            "pairwise_l2": pairwise_l2,
            "std_loss": std_loss
        })
    return loss


def stability_train_loop(dataset_name, loss_fn_name, tokenizer, dataloaders, num_epochs, model, optimizer, lr_scheduler, steps_to_eval):
    eval_datas = []
    # eval_data = trainability_eval(
    #     model, tokenizer, dataloaders, 0, 0, dataset_name
    # )
    # eval_datas.append(eval_data)
    progress_bar = tqdm(range(num_epochs * len(dataloaders['train'])), desc="Training")
    step = 0
    for epoch in range(num_epochs):
        
        for train_batch in dataloaders['train']:
            progress_bar.update(1)
            step += 1
            model.train()
            with accelerator.accumulate(model):
                loss = None
                if loss_fn_name == "min_loss":
                    outputs = model(**train_batch, labels=train_batch['input_ids'])
                    loss = outputs.loss
                print(loss)
                accelerator.backward(loss)
                        
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # eval_data = trainability_eval(
        #     model, tokenizer, dataloaders, epoch, step, dataset_name
        # )
        # eval_datas.append(eval_data)
    return model, eval_datas
