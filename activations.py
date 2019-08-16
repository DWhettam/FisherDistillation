from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import genotypes
import json
import argparse
from pytorchcv.model_provider import get_model as ptcv_get_model
import os
from models import *
from models.darts import NetworkCIFAR as Network
from tqdm import tqdm
import re


os.mkdir('checkpoints/') if not os.path.isdir('checkpoints/') else None

parser = argparse.ArgumentParser(description='Student/teacher training')
#parser.add_argument('arch', choices=['darts', 'dense', 'wrn'], type=str, help='Learn a teacher or a student')
parser.add_argument('--cifar_loc', default='/home/dan/Desktop/Dissertation/xdistill/data', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--cifar_loc', default='/disk/scratch/s1874193/datasets/cifar', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--cifar_loc', default='/afs/inf.ed.ac.uk/user/s18/s1874193/Desktop/', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--checkpoint', '-s', default='dartsteach_dartsstudent.student', type=str, help='checkpoint to save/load student')

if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

args = parser.parse_args()
print(args)

class ReturnLayers(nn.Module):
    def __init__(self, model):
        super(ReturnLayers, self).__init__()
        self.model = model
    def forward(self, x):
        activations = []
        for i, module in enumerate(self.model.features._modules.values()):
            x = module(x)
            if i % 2 == 0:
                activations.append(x)
        x = x.view(x.size(0), -1)
        x = self.model.output(x)
        return x, activations

def get_cifar_loaders(cifar_loc, batch_size=128, workers=0):
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                        train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                       train=False, download=False, transform=transform_validate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False,
                                             num_workers=workers)
    return trainloader, valloader

def load_network(file, arch):
    if arch == 'dense':
        net_checkpoint = torch.load("/home/dan/Desktop/Dissertation/xdistill/checkpoints/complete/%s" % file)
        net = ptcv_get_model("densenet100_k12_bc_cifar10", pretrained=False)
        net = ReturnLayers(net).cuda()
        net.load_state_dict(net_checkpoint['state'])
    elif arch == 'wrn':
        net_checkpoint = torch.load("/home/dan/Desktop/Dissertation/xdistill/checkpoints/complete/%s" % file)
        start_epoch = net_checkpoint['epoch']
        net = WideResNet(net_checkpoint['depth'], net_checkpoint['width'], num_classes=net_checkpoint['num_classes'], dropRate=0).cuda()
        net.load_state_dict(net_checkpoint['state'])
    elif arch == 'darts' or arch == 'dart' :
        net_checkpoint = torch.load("/home/dan/Desktop/Dissertation/xdistill/checkpoints/complete/%s" % file)
        genotype = eval("genotypes.DARTS")
        net = Network(16, 10, 10, True, genotype).cuda()
        net.load_state_dict(net_checkpoint['state'])
        net.drop_path_prob = 0.2
    return net

if __name__ == "__main__":
    trainloader, _ = get_cifar_loaders(args.cifar_loc, 1)
    dataiter = iter(trainloader)
    data = torch.rand(1,3,32,32).cuda()
    input, target = dataiter.next()
    input, target = input.cuda(), target.cuda()

    for filename in os.listdir('/home/dan/Desktop/Dissertation/xdistill/checkpoints/complete'):
      arch = re.search(r'_(.*?)student', filename).group(1)
      #net = load_network('denseteach_dartstudent.student.t7', 'darts')
      net = load_network(filename, arch)
      net.eval()

      out, act = net(input)
      act = act[0]
      act = F.normalize(act.pow(2).mean(1))#.view(act.size(0), -1))
      torchvision.utils.save_image(act, '/home/dan/Desktop/Dissertation/xdistill/activations/%s_activations.png' % filename)
