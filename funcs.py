import torch
import torch.nn.functional as F
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision
import time
from functools import reduce
from models import *
import random
import time
import operator
import torchvision
import torchvision.transforms as transforms


def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_no_params(net, verbose=True):
    params = net.state_dict()
    tot = 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        if 'bn' not in p:
            if verbose:
                print('%s has %d params' % (p, no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)
    return tot


def find(input):
    # Find as in MATLAB to find indices in a binary vector
    return [i for i, j in enumerate(input) if j]


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_inf_params(net, verbose=True, sd=False):
    if sd:
        params = net
    else:
        params = net.state_dict()
    tot = 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()

        if ('num_batches_tracked' not in p) and ('running' not in p) and ('mask' not in p):
            tot += no

            if verbose:
                print('%s has %d params' % (p, no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)

    return tot


class Fisher_grow:
    def __init__(self, module_name='Cell'):
        self.module_name = module_name
        self.masks = []

    def go_fish(self, model):
        self._get_fisher(model)
        tot_loss = self.fisher#.div(1) + 1e6 * (1 - self.masks) #giga
        return tot_loss

    def _get_fisher(self, model):
        masks=[]
        fisher=[]
        flops=[]

        # self._update_flops(model)

        for m in model.modules():
            if m._get_name() == 'Cell':
              masks.append(m.mask.detach())
              fisher.append(m.running_fisher.detach())

              m.reset_fisher()


        self.masks = self.concat(masks)
        self.fisher = self.concat(fisher)

    def _get_masks(self, model):
        masks=[]

        for m in model.modules():
            if m._get_name() == 'Cell':
              masks.append(m.mask.detach())


        self.masks = self.concat(masks)

    @staticmethod
    def concat(input):
        return torch.cat([item for item in input])

def tile(a, dim, n_tile):
  init_dim = a.size(dim)
  repeat_idx = [1] * a.dim()
  repeat_idx[dim] = n_tile
  a = a.repeat(*(repeat_idx))
  order_index = torch.cuda.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
  return torch.index_select(a, dim, order_index)


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

def get_cifar100_loaders(cifar_loc='../cifar100', batch_size=128, workers=0):
    num_classes = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=cifar_loc,
                                            train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root=cifar_loc,
                                           train=False, download=False, transform=transform_validate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers)
    return trainloader, valloader

def get_imagenet_loaders(imagenet_loc, batch_size=128, workers=12):
    num_classes = 1000
    traindir = os.path.join(imagenet_loc, 'train')
    valdir = os.path.join(imagenet_loc, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_validate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
    valset = torchvision.datasets.ImageFolder(valdir, transform_validate)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers,
                                              pin_memory = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    return trainloader, valloader


def one_shot_fisher(net, trainloader, n_steps=1, cuda=False):
    params = get_no_params(net, verbose=False)
    criterion = nn.CrossEntropyLoss()
    # switch to train mode
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    dataiter = iter(trainloader)
    fisher_grow = Fisher_grow()
    data = torch.rand(net.input_spatial_dims)
    if cuda:
        data = data.cuda()
    net(data)
    fisher_grow._get_masks(net)

    NO_STEPS = n_steps # single minibatch
    for i in range(0, NO_STEPS):
        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        if cuda:
            input, target = input.cuda(), target.cuda()

        # compute output
        output, act = net(input)

        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fisher_inf = fisher_grow.go_fish(net)
    fish_market = dict()
    running_index = 0

    block_count = 0
    data = torch.rand(1,16,32,32)
    if cuda:
        data = data.cuda()
    # partition fisher_inf into blocks blocks blocks
    for m in net.modules():
        if m._get_name() == 'Cell':
          mask_indices = range(running_index, running_index + len(m.mask))
          fishies = [fisher_inf[j] for j in mask_indices]
          running_index += len(m.mask)

          fish = sum(fishies)

          fish_market[block_count] = fish
          block_count +=1


    return params, fish_market
