import torch
from funcs import *
from models import *
import genotypes
#import matplotlib.pyplot as plt
#import numpy as np
from tqdm import tqdm
from statistics import mean, median
from models.darts import NetworkCIFAR as Network
import argparse
from operations import *


parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset',         type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')
#parser.add_argument('--data_loc', default='/disk/scratch/s1874193/datasets/cifar', type=str, help='folder containing cifar train and val folders')
parser.add_argument('--data_loc', default='/afs/inf.ed.ac.uk/user/s18/s1874193/Desktop/', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--data_loc', default='/home/dan/Desktop/Dissertation/xdistill/data', type=str, help='folder containing cifar train and val folders')
parser.add_argument('--batch_size',    default=32, type=int)
parser.add_argument('--workers',       default=0, type=int)
parser.add_argument('--param_goal',    default=1400000, type=int)
parser.add_argument('--top_n',         default=1, type=int)
parser.add_argument('--continue_from', default=0, type=int)
parser.add_argument('--generate_random', action='store_true')
parser.add_argument('--naive', action='store_true')
parser.add_argument('--save_file',     default='darts_uniform_teach')
parser.add_argument('--init_channels', default=16, type=int, help='initial number of channels')
parser.add_argument('--base_model', default='darts_fisher', type=str, help='basemodel')
parser.add_argument('--std_arch', type=str, default='DARTS', help='which student architecture to use')
parser.add_argument('--auxiliary', action='store_true')

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

def update_model(model, convs):
    convs = str(convs)[1:-1].rstrip().replace(',', '')
    cs = convs.replace('\n', ' ').split()
    prefix = len("'models.blocks.")
    suffix = len("'>")

    r = [w for w in cs if '<class' not in w]
    blocks = [c[prefix:-suffix] for c in r]
    convs  = [string_to_conv[c] for c in blocks]

    for i, c in enumerate(convs):
        model = update_block(i, model, c)

    return model, blocks

def generate_random():
    cifar_random_search(args.save_file)


def rank_at_param_goal(std_arch, init_channels, num_classes, layers, auxiliary, data_loc, batch_size, continue_from, dataset, param_goal, save_file):
    var = 0.025 * param_goal
    lower_bound = param_goal - var
    upper_bound = param_goal + var
    #models = master[master['params'].between(lower_bound, upper_bound, inclusive=True)]

    if dataset == 'cifar10':
        data = torch.rand(2,3,32,32).to(device)
        genotype = eval("genotypes.%s" % std_arch)
        student = Network(init_channels, num_classes, layers, auxiliary, genotype).to(device)
        student.drop_path_prob = 0.2
        student(data)
        train, val = get_cifar_loaders(data_loc, batch_size)
    elif dataset == 'cifar100':
        data = torch.rand(1,3,32,32).to(device)
        genotype = eval("genotypes.%s" % std_arch)
        student = Network(init_channels, num_classes, layers, auxiliary, genotype).to(device)
        student.drop_path_prob = 0.2
        student(data)
        train, val = get_cifar100_loaders(data_loc)
    elif dataset == 'imagenet':
        data = torch.rand(1,3,224,224).to(device)
        genotype = eval("genotypes.%s" % std_arch)
        student = Network(init_channels, num_classes, layers, auxiliary, genotype).to(device)
        student.drop_path_prob = 0.2
        student(data)
        train, val = get_imagenet_loaders(data_loc)

    for i in tqdm(range(continue_from, continue_from+100)):
        if not args.naive:
          params, fish = one_shot_fisher(student, train, 1)
          print("Model size: " + str(params))
          if params < lower_bound:
            sorted_fish = sorted(fish.items(), key=lambda kv: kv[1], reverse=True)
            for fish in sorted_fish:
                if student.update_cell(fish[0]):
                    break
        else:
            for idx, cell in enumerate(student.cells):
                student.update_cell((len(student.cells)-1) - idx)
                params = get_no_params(student, verbose=False)
                if params > lower_bound:
                    break


        if params > lower_bound:
          filename = 'checkpoints/%s.t7' % (save_file)
          save_checkpoint({
                  'state': student.state_dict(),
                  'depth': 10,
                  'dataset': dataset,
                  'num_classes': 10,
          }, filename=filename)
          print("Model reached valid size: " + str(sum(p.numel() for p in student.parameters())) + " params")
          print("Saving")

          for cell in student.cells:
              for op in cell._ops:
                  if isinstance(op, SepConv) or isinstance(op, DilConv):
                    f= open("darts_teach_groups.txt","a")
                    f.write(str(op._groups) + "\n")
                    f.close()

          return student



if __name__ == "__main__":
  args = parser.parse_args()
  if args.generate_random:
      generate_random()
  else:
      rank_at_param_goal(args.std_arch, args.init_channels, 10, 10, args.auxiliary, args.data_loc, args.batch_size, args.continue_from, args.dataset, args.param_goal, args.save_file)
