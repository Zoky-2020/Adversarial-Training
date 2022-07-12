import os
import argparse
import torchvision
from torchvision import transforms
import torch.optim as optim
import numpy as np
import torch
from torch import nn
import datetime
from models import *
from utils.save import save_checkpoint
import attack


parser = argparse.ArgumentParser(description="Standard Adversarial Training")
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=7, help='random seed')

parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="True: rand | False: none")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")

parser.add_argument('--net', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./ckpt/SAT', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--device',type=str,default='1',help='gpu id')

parser.add_argument('--attack_eps',type=float, default=0.031, help='for testing')
parser.add_argument('--attack_step_size',type=float, default=0.003, help='for testing')
parser.add_argument('--attack_steps',type=int, default=20, help='for testing')
parser.add_argument('--attack_rand',type=str, default='none', help='for testing, trades | rand | none')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.device

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def train(model,train_loader,optimizer):
    loss_sum = 0
    start_time = datetime.datetime.now()
    for imgs,labels in train_loader:
        imgs,labels = imgs.cuda(),labels.cuda()
        imgs_adv = attack.pgd(model,imgs,labels,args.epsilon,args.num_steps,args.step_size,args.rand_init)
        model.train()
        optimizer.zero_grad()
        output = model(imgs_adv)
        loss = nn.CrossEntropyLoss(reduction='mean')(output,labels)
        loss_sum +=loss.item()
        loss.backward()
        optimizer.step()
    end_time = datetime.datetime.now()
    time = (end_time - start_time).seconds
    return time,loss_sum


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 0.5 * args.epochs:
        lr = args.lr * 0.1
    if epoch >= 0.75 * args.epochs:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),])
transform_test = transforms.Compose([
    transforms.ToTensor(),])

print('==> Load Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

print('==> Load Model')
if args.net == "resnet18":
    model = ResNet18().cuda()
if args.net == "WRN":
    model = Wide_ResNet(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
print(args.net)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


lr_recording = ''
print('==> Standard Adversarial Traininig')
for epoch in range(0,args.epochs):
    # adjust_learning_rate(optimizer,epoch+1)
    adjust_learning_rate(optimizer, epoch + 1)
    train_time, train_loss = train(model, train_loader, optimizer)
    nat_loss, nat_acc = attack.clean_acc(model, test_loader)
    fgsm_loss, fgsm_acc = attack.fgsm_acc(model, test_loader)
    pgd_loss, pgd_acc = attack.pgd_acc(model, test_loader, args.attack_eps, args.attack_steps, args.attack_step_size, args.attack_rand)

    print(
        'Epoch: {}|{} | Train Time: {:.2f} s | Natural Test Acc {:.2f} | FGSM Test Acc {:.2f} | PGD20 Test Acc {:.2f} |'.format(
        epoch + 1, args.epochs, train_time, nat_acc, fgsm_acc, pgd_acc))

    save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), }, args.out_dir)
