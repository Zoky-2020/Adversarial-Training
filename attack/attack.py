import torch
from torch import nn
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable

def fgsm(model,data,label,target=False,alpha=0.031,rand_init=False):
    model.eval()
    if rand_init:
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-alpha,alpha,data.shape)).float().cuda()
        x_adv = torch.clamp(x_adv,0.0,1.0)
    else:
        x_adv = data.detach()
    x_adv.requires_grad_()
    output = model(x_adv)
    model.zero_grad()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss(reduction="mean")(output,label)
        if target:
            loss = -loss
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + alpha*grad_sign
    x_adv = torch.min(torch.max(x_adv,data-alpha), data + alpha)
    x_adv = torch.clamp(x_adv,0.0,1.0)
    return x_adv


def pgd(model,data,label,epsilon,num_steps,step_size,rand_init):
    model.eval()
    if rand_init == "trades":
        x_adv = data.detach() + 0.001* torch.randn(data.shape).cuda().detach()
    elif rand_init == "rand":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon,epsilon,data.shape)).float().cuda()
        x_adv = torch.clamp(x_adv,0.0,1.0)
    else:
        x_adv = data.detach()
     
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="mean")(output,label)
        loss.backward()
        perturbation = step_size* x_adv.grad.sign()
        x_adv = x_adv.detach() + perturbation
        x_adv = torch.min(torch.max(x_adv,data - epsilon),data + epsilon)
        x_adv = torch.clamp(x_adv,0.0,1.0)
    return x_adv

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def cw_attack(model, data, label, epsilon, num_steps, step_size, rand_init):
    model.eval()
    if rand_init == "trades":
        x_adv = data.detach() + 0.001* torch.randn(data.shape).cuda().detach()
    elif rand_init == "rand":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon,epsilon,data.shape)).float().cuda()
        x_adv = torch.clamp(x_adv,0.0,1.0)
    else:
        x_adv = data.detach()
    
    for k in range(num_steps):
        x_adv.requires_grad_()
        outputs = model(x_adv)
        with torch.enable_grad():
            loss_adv = cwloss(outputs, label)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv
