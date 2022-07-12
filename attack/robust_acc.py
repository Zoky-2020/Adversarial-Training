import numpy  as np
import torch
from torch import nn 
import attack

def clean_acc(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(),label.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output,label).item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    return test_loss,test_acc

def fgsm_acc(model,test_loader,alpha=0.031, rand_init=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            x_adv = attack.fgsm(model,data,label,alpha=alpha,rand_init=rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, label).item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /=len(test_loader.dataset)
    fgsm_acc = correct / len(test_loader.dataset)
    return test_loss, fgsm_acc

def pgd_acc(model,test_loader,epsilon,num_steps,step_size,rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data,label in test_loader:
            data, label = data.cuda(), label.cuda()
            x_adv = attack.pgd(model,data,label,epsilon,num_steps,step_size,rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, label).item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /=len(test_loader.dataset)
    pgd_acc = correct / len(test_loader.dataset)
    return test_loss, pgd_acc


def cw_acc(model, test_loader, epsilon, num_steps,step_size, rand_init):
    model.eval()

    test_loss = 0.0
    correct = 0
    with torch.enable_grad():
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.cuda(), label.cuda()
            x_adv = attack.cw_attack(model,data,label, epsilon, num_steps, step_size, rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, label).item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /=len(test_loader.dataset)
    cw_acc = correct / len(test_loader.dataset)
    return test_loss, cw_acc
