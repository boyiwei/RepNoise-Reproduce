from immunization_llms.datasets import DEVICE
import torch
from torch import nn
import torch.nn.functional as F

from loguru import logger

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, xy_only=False, xxyy_only=False):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.xy_only = xy_only
        self.xxyy_only = xxyy_only
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
 
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
 
    def forward(self, source, target, xy_only=False):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # logger.info(f"MMD Kernel: XX: {torch.mean(XX)}, YY: {torch.mean(YY)}, XY: {torch.mean(XY)}, YX: {torch.mean(YX)}")
        if xy_only:
            return torch.mean(XY)
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def cross_entropy_loss(
    lm_logits,
    labels
):
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class GaussianKernelCovarianceLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianKernelCovarianceLoss, self).__init__()
        self.sigma = sigma

    def gaussian_kernel(self, x, y):
        # Compute pairwise squared Euclidean distances
        distances = torch.sum((x.unsqueeze(1) - y.unsqueeze(0))**2, dim=2)
        return torch.exp(-distances / (2 * self.sigma**2))

    def forward(self, features_x, features_y):
        kernel_matrix = self.gaussian_kernel(features_x, features_y)
        
        mean_x = torch.mean(kernel_matrix, dim=0)
        mean_y = torch.mean(kernel_matrix, dim=1)
        
        covariance = torch.mean(kernel_matrix) - torch.mean(mean_x) * torch.mean(mean_y)
        
        return -covariance


def decorrelation_loss(y_pred):
    """
    Decorrelation loss for a batch of samples.
    
    Args:
    - y_pred (Tensor): Predicted representations, shape [batch_size, representation_dim]
    
    Returns:
    - decor_loss (Tensor): Decorrelation loss
    """
    y_pred_normalized = y_pred / torch.norm(y_pred, dim=1, keepdim=True)
    
    N = y_pred_normalized.size(0)
    pairwise_dot_products = torch.matmul(y_pred_normalized, y_pred_normalized.t())
    decor_loss = torch.sum(pairwise_dot_products ** 2) - torch.sum(torch.diag(pairwise_dot_products) ** 2)
    
    return decor_loss / (N * (N - 1))


def reverse_cross_entropy_loss(input, target):
    # Convert target to one-hot encoding
    shift_logits = input[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    input = shift_logits.view(-1, shift_logits.size(-1))
    target = shift_labels.view(-1)
    target_one_hot = F.one_hot(target, num_classes=input.size(-1))

    # invert the one hot so its 1 when the target is 0 and 0 when the target is 1
    target_one_hot = 1 - target_one_hot
    # uniform probability for all the other classes
    target_one_hot = target_one_hot * 1/(input.size(-1)-1)

    # Compute softmax
    softmax = torch.softmax(input, dim=-1)

    # Compute log softmax
    log_softmax = torch.log(softmax)

    # Compute cross-entropy loss
    loss = -torch.sum(target_one_hot * log_softmax) / input.size(0)

    return loss


from torch.autograd import Variable



def _find_z(model, inputs, targets, h):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()

    loss_z = cross_entropy_loss(model.eval()(inputs_embeds=inputs).logits, targets)                

    loss_z.backward()

    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None]+1e-7)  
    inputs.grad.detach_()
    inputs.grad.zero_()
    model.zero_grad()

    return z, norm_grad
    
# linearly increase h to 1.5 over the course of time
# lambda is at 4
def regularizer(model, inputs, targets, h = 3., lambda_ = 4):
    '''
    Regularizer term in CURE
    '''
    inputs=Variable(model.get_input_embeddings().weight[inputs].clone())
    # track gradient of token embeddings
    inputs.requires_grad=True
    z, norm_grad = _find_z(model, inputs, targets, h)
    inputs.requires_grad_()
    outputs_pos = model.eval()(inputs_embeds=inputs + z)
    outputs_orig = model.eval()(inputs_embeds=inputs)

    loss_pos = cross_entropy_loss(outputs_pos.logits, targets)
    loss_orig = cross_entropy_loss(outputs_orig.logits, targets)
    grad_diff = torch.autograd.grad(
        (loss_pos-loss_orig), 
        inputs, # grad_outputs=torch.ones(targets.size()).to(device),
        create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()

    return torch.sum(lambda_ * reg) / float(inputs.size(0)), norm_grad


def masked_token_ce_loss(
    logits,
    labels,
    mask
):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # apply mask
    # shifted masks
    shift_logit_mask = mask[..., :-1].contiguous()
    expanded_mask = shift_logit_mask.unsqueeze(-1).expand(-1, -1, shift_logits.size(-1))
    shift_label_mask = mask[..., 1:].contiguous()
    shift_logits = shift_logits * expanded_mask
    shift_labels = shift_labels * shift_label_mask
    shift_labels[shift_labels == 0] = -100
    shift_labels = shift_labels.type(torch.LongTensor)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(DEVICE), shift_labels.view(-1).to(DEVICE))
    return loss

def l2_distance(source, target):
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    return torch.mean(L2_distance)

def l2_pairwise(batch_tensor):
    # Expand dimensions for broadcasting
    expanded_batch = batch_tensor.unsqueeze(1)
    
    # Compute pairwise differences
    pairwise_diff = expanded_batch - batch_tensor
    
    # Compute L2 distance
    l2_distances = torch.sqrt(torch.sum(pairwise_diff ** 2, dim=2) + 1e-8)
    
    return l2_distances.mean()

def pairwise_covariance(batch_tensor):
    """
    Compute pairwise covariance between items in a batch of tensors.
    
    Args:
    - batch_tensor (torch.Tensor): Batch of tensors with shape (batch_size, features)
    
    Returns:
    - covariances (torch.Tensor): Pairwise covariances with shape (batch_size, batch_size)
    """
    
    # Compute mean of batch
    mean = batch_tensor.mean(dim=0)
    
    # Compute pairwise differences
    pairwise_diff = batch_tensor - mean
    
    # Compute pairwise covariances
    covariances = torch.mm(pairwise_diff, pairwise_diff.t())
    # only the off-diagonal elements
    covariances = covariances - torch.diag(covariances.diag())
    return covariances.mean()

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
