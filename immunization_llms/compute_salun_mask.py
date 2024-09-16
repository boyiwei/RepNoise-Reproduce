import os
from immunization_llms.loss import masked_token_ce_loss
import torch
from tqdm import tqdm
from immunization_llms.datasets import DEVICE, accelerator
from transformers import (
    AutoModelForCausalLM
)
from loguru import logger

def construct_gradient_mask(
    dataloaders,
    model_name,
    learning_rate
):
    dtype = "auto"
    if 'meta-llama' in model_name:
        dtype = torch.bfloat16
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer  = accelerator.prepare(
        model, optimizer
    )
    # prepare dataloaders dict
    for k, v in dataloaders.items():
        dataloaders[k] = accelerator.prepare(v)

    logger.info("Beginning mask training loop")
    gradients = {}

    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0
    progress_bar = tqdm(range(len(dataloaders['harmful'])), desc="Training")
    step = 0
    for harmful_loss, harmless_batch in zip(dataloaders['harmful'], dataloaders['harmless']):
        progress_bar.update(1)
        step += 1
        
        # compute output
        with accelerator.accumulate(model):
            harmful_outputs = model(harmful_loss['input_ids'], attention_mask=harmful_loss['attention_mask'])
            mask = ~torch.eq(harmful_loss['input_ids'], harmless_batch['input_ids'])
            # Convert the mask to float
            mask = mask.float().to(DEVICE)
            loss = -masked_token_ce_loss(
                harmful_outputs.logits,
                harmful_loss['input_ids'],
                mask
            )
            logger.info(f"Loss: {loss}")
            optimizer.zero_grad()
            accelerator.backward(loss)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])
    
    # # save gradients as pickle
    # import pickle
    # with open('gradients.pickle', 'wb') as handle:
    #     pickle.dump(gradients, handle, protocol=pickle.HIGHEST_PROTOCOL)

    hard_dict = {}

    # Concatenate all tensors into a single tensor
    all_elements = torch.cat([tensor.flatten().to(DEVICE) for tensor in gradients.values()])
    mean = torch.mean(all_elements).to('cpu')

    for key, tensor in tqdm(gradients.items()):
    # construct the mask if the gradient is greater than the threshold
    # mask is 1 and 0s
        mask = (tensor.to('cpu') > mean).float().type(torch.uint8)
        hard_dict[key] = mask
        
    torch.save(hard_dict, os.path.join('models', "{}_salun_mask.pt".format(model_name.replace('/', '_'))))