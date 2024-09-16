from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import Trainer, TrainingArguments
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttention,
    GPTNeoFlashAttention2,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
)
from immunization_llms.datasets import accelerator
from transformers.models.opt.modeling_opt import OPTAttention, OptFlashAttention2
from transformers.models.phi.modeling_phi import PhiAttention, PhiFlashAttention2

# TODO set this appropriately for your models attention modules
ATTENTION_TYPES = [
    LlamaAttention,
    LlamaFlashAttention2,
    OPTAttention,
    OptFlashAttention2,
    PhiAttention,
    PhiFlashAttention2,
    GPTNeoAttention,
    GPTNeoFlashAttention2,
]


def vaccine_training_step(model: nn.Module, inputs, RHO=0.1) -> torch.Tensor:
    model.train()
    epsilon = compute_epsilon(model, inputs, RHO)
    loss = parameter_gradients_with_perturbation(model, inputs, epsilon)
    return loss

def compute_epsilon(
    model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], rho
):
    with ComputeRepresentationPerturbation(model, rho) as rp:
        sft_training_step(model, inputs)
    return rp.get_epsilon()

def parameter_gradients_with_perturbation(
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    epsilon: Dict,
) -> torch.Tensor:
    with ApplyPerturbation(model, epsilon):
        loss = sft_training_step(model, inputs)
    return loss.detach() # type: ignore

def sft_training_step(
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
) -> Tuple[torch.Tensor, Any]:
    """Similar to Trainer.training_step."""
    loss = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids']).loss

    accelerator.backward(loss)

    loss = loss.detach() # type: ignore
    return loss


class HookContextManager:
    def __init__(self):
        self.hooks = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_attention_blocks(self, model: nn.Module) -> List[nn.Module]:
        if isinstance(model, DistributedDataParallel):
            model = model.module
        module_list = []
        for name, module in model.named_modules():
            if is_attention_layer(name, module):
                module_list += [module]
        return module_list


class ComputeRepresentationPerturbation(HookContextManager):
    SMALL_CONSTANT = 1e-7

    def __init__(
        self,
        model: nn.Module,
        rho: float,
    ):
        super().__init__()
        self.model = model
        self.epsilon = {}
        self.rho = rho
        self.grad_norm: float

    def get_epsilon(self):
        return self.epsilon

    def get_gradient_norm(self):
        return self.grad_norm

    def track_gradient_hook(self, module: nn.Module, grad_output: Tuple[torch.Tensor]):
        grad = grad_output[0].detach().clone()
        self.epsilon[module] = grad

    def __enter__(self):
        for layer in self.get_attention_blocks(self.model):
            self.epsilon[layer] = 0
            hook = layer.register_full_backward_pre_hook(self.track_gradient_hook)  # type: ignore
            self.hooks.append(hook)
        return self

    def _grad_norm(self) -> float:
        return (
            sum(torch.norm(grad).to(accelerator.device) ** 2 for grad in self.epsilon.values()) ** 0.5
        ).item()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.remove_hooks()

        self.grad_norm = self._grad_norm()
        for module, grad in self.epsilon.items():
            scale = self.rho / (self.grad_norm + self.SMALL_CONSTANT)
            new_eps = grad * scale
            self.epsilon[module] = new_eps.detach().clone()


class ApplyPerturbation(HookContextManager):
    def __init__(
        self,
        model: nn.Module,
        epsilon: Dict[nn.Module, torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.epsilon = epsilon

    def perturbation_hook(self, module, input, output):
        perturbation = self.epsilon[module]
        output[0].data = output[0] + perturbation
        return output

    def __enter__(self):
        for layer in self.get_attention_blocks(self.model):
            hook = layer.register_forward_hook(self.perturbation_hook)
            self.hooks.append(hook)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.remove_hooks()


def is_attention_layer(name: str, module: nn.Module):
    attention_string = "self_attn"
    if name[-len(attention_string) :] == attention_string:
        return True
    return any(isinstance(module, attention_type) for attention_type in ATTENTION_TYPES)