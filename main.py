''''Writing everything into one script..'''
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
import os
import types
from tensorboardX import SummaryWriter
import time
from funcs import *
from operations import *
from models import *
from model import NetworkCIFAR as Network
from tqdm import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.autograd import Variable
from fisher_rank import rank_at_param_goal

os.mkdir('checkpoints/') if not os.path.isdir('checkpoints/') else None

parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'],
                    help='Choose between Cifar10/100/imagenet.')
parser.add_argument('mode', choices=['student', 'teacher'], type=str, help='Learn a teacher or a student')
parser.add_argument('--imagenet_loc', default='/disk/scratch_ssd/imagenet', type=str,
                    help='folder containing imagenet train and val folders')
parser.add_argument('--cifar_loc', default='/disk/scratch/s1874193/datasets/cifar', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--cifar_loc', default='/afs/inf.ed.ac.uk/user/s18/s1874193/Desktop/', type=str, help='folder containing cifar train and val folders')
#parser.add_argument('--cifar_loc', default='/home/dan/Desktop/Dissertation/xdistill/data', type=str, help='folder containing cifar train and val folders')
parser.add_argument('--workers', default=2, type=int, help='No. of data loading workers. Make this high for imagenet')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--student_checkpoint', '-s', default='wrn_40_2_student_KT', type=str, help='checkpoint to save/load student')
parser.add_argument('--teacher_checkpoint', '-t', default='darts_teach', type=str, help='checkpoint to load in teacher')
parser.add_argument('--teacher_arch', choices=['wrn', 'densenet', 'darts'])
parser.add_argument('--student_arch', choices=['wrn', 'densenet', 'darts'])
parser.add_argument('--fisher', action='store_true', default=False, help='training model for fisher expansion')
parser.add_argument('--fisher_teacher', action='store_true', default=False, help='usingf fisher expanded model as a teacher')
parser.add_argument('--groups_file', default='darts_teach_groups.txt', type=str, help='file containing info on updated darts groups')
parser.add_argument('--param_goal', default=1400000, type=int)

# network stuff
parser.add_argument('--student_depth', default=10, type=int, help='student depth')
parser.add_argument('--student_width', default=2, type=float, help='student width')
parser.add_argument('--teach_depth', default=25, type=int, help='teacher depth')
parser.add_argument('--teach_width', default=2, type=float, help='teacher width')
parser.add_argument('--module', default=None, type=str,
                    help='path to file containing custom Conv and maybe Block module definitions')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--std_arch', type=str, default='DARTS', help='which student architecture to use')

# learning stuff
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
parser.add_argument('--alpha', default=0.0, type=float, help='alpha for KD')
parser.add_argument('--aux_loss', default='AT', type=str, help='AT or SE loss')
parser.add_argument('--beta', default='[1000, 1000, 1000]', type=str, help='beta vector for AT')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--batch_size', default=64, type=int,
                    help='minibatch size')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

#overloads nets to return activations in forward()
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




def create_optimizer(lr, net):
    print('creating optimizer with lr = %0.5f' % lr)
    return torch.optim.SGD(net.parameters(), lr, 0.9, weight_decay=args.weightDecay)


def train_teacher(net):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def train_student(net, teach):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    teach.eval()

    end = time.time()

    student_total_params = sum(p.numel() for p in net.parameters())
    teach_total_params = sum(p.numel() for p in teach.parameters())
    print("student params: " + str(student_total_params))
    print("teach params: " + str(teach_total_params))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs_student, ints_student = net(inputs)
        outputs_teacher, ints_teacher = teach(inputs)

        # If alpha is 0 then this loss is just a cross entropy.

        if args.alpha > 0:
            loss = distillation(outputs_student, outputs_teacher, targets, args.temperature, args.alpha)
        else:
            loss = criterion(outputs_student, targets)
        # Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        # paper) and adjust the beta term accordingly.

        for i in range(len(ints_student)):
            loss += BETA[i] * aux_loss(ints_student[i], ints_teacher[i])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs_student.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def validate(net, checkpoint=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(valloader):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

            losses.update(loss.item(), inputs.size(0))
            top1.update(err1[0], inputs.size(0))
            top5.update(err5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(valloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))


    out = (' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(out)

    val_losses.append(losses.avg)
    val_errors.append(top1.avg)

    if checkpoint:
        print('Saving..')
        state = {
            'state': net.state_dict(),
            'epoch': epoch,
            'width': args.width,
            'depth': args.depth,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'val_losses': val_losses,
            'val_errors': val_errors,
            'dataset': args.dataset,
            'num_classes': num_classes,
        }
        print('SAVED!')
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)


if __name__ == '__main__':
    # Stuff happens from here:
  if args.aux_loss == 'AT':
      aux_loss = at_loss
  else:
      raise NotImplementedError('No other aux losses implemented')

  print(vars(args))
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

  val_losses = []
  train_losses = []
  val_errors = []
  train_errors = []

  best_acc = 0
  start_epoch = 0
  epoch_step = json.loads(args.epoch_step)
  BETA = json.loads(args.beta)
  print(BETA)
  time.sleep(4)

  # Data and loaders
  print('==> Preparing data..')
  if args.dataset == 'cifar10':
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
      trainset = torchvision.datasets.CIFAR10(root=args.cifar_loc,
                                              train=True, download=False, transform=transform_train)
      valset = torchvision.datasets.CIFAR10(root=args.cifar_loc,
                                            train=False, download=False, transform=transform_validate)
  elif args.dataset == 'cifar100':
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
      trainset = torchvision.datasets.CIFAR100(root=args.cifar_loc,
                                               train=True, download=False, transform=transform_train)
      valset = torchvision.datasets.CIFAR100(root=args.cifar_loc,
                                                  train=False, download=False, transform=transform_validate)

  elif args.dataset == 'imagenet':
      num_classes = 1000
      traindir = os.path.join(args.imagenet_loc, 'train')
      valdir = os.path.join(args.imagenet_loc, 'val')
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

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True if args.dataset == 'imagenet' else False)
  valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                          num_workers=args.workers,
                                          pin_memory=True if args.dataset == 'imagenet' else False)

  criterion = nn.CrossEntropyLoss()


  def load_network(teacher=True):
      if args.teacher_arch == 'densenet' and teacher :
          net = ptcv_get_model("densenet40_k36_bc_cifar10", pretrained=True)
          net = ReturnLayers(net).to(device)
      elif args.teacher_arch == 'wrn' and teacher :
          net_checkpoint = torch.load('checkpoints/wrn_40_2_T.t7')
          start_epoch = net_checkpoint['epoch']
          net = WideResNet(net_checkpoint['depth'], net_checkpoint['width'], num_classes=net_checkpoint['num_classes'], dropRate=0).to(device)
          net.load_state_dict(net_checkpoint['state'])
      elif args.teacher_arch == 'darts' and teacher and not args.fisher and not args.fisher_teacher:
          net_checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
          genotype = eval("genotypes.%s" % args.std_arch)
          net = Network(args.init_channels, num_classes, args.teach_depth, args.auxiliary, genotype).to(device)
          net.load_state_dict(net_checkpoint['state'])
      elif args.teacher_arch == 'darts' and teacher and args.fisher:
          net = rank_at_param_goal(args.std_arch, args.init_channels, 10, 10, args.auxiliary, \
                                   args.cifar_loc, args.batch_size, 100, args.dataset, args.param_goal, args.teacher_checkpoint)
      elif args.teacher_arch == 'darts' and teacher and args.fisher_teacher:
          net_checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
          genotype = eval("genotypes.%s" % args.std_arch)
          net = Network(args.init_channels, num_classes, args.student_depth, args.auxiliary, genotype).to(device)
          with open(args.groups_file, 'r+') as f:
            groups = f.readlines()
            group = 0
            for cell in net.cells:
                for op in cell._ops:
                    if isinstance(op, SepConv) or isinstance(op, DilConv):
                      op.update(int(groups[group]))
                      group +=1
          net.drop_path_prob = 0.2
          data = torch.rand(2,3,32,32).cuda()
          net(data)
          net.load_state_dict(net_checkpoint['state'])

      elif args.student_arch == 'densenet' and not teacher :
          net = ptcv_get_model("densenet100_k12_bc_cifar10", pretrained=False)
          net = ReturnLayers(net).to(device)
      elif args.student_arch == 'wrn' and not teacher :
          net = WideResNet(args.student_depth, args.student_width, num_classes=num_classes, dropRate=0).to(device)
      elif args.student_arch == 'darts' and not teacher:
          genotype = eval("genotypes.%s" % args.std_arch)
          net = Network(args.init_channels, num_classes, args.student_depth, args.auxiliary, genotype).to(device)
          net.drop_path_prob = 0.2
      return net

  if args.mode == 'teacher':

      if args.resume:
          print('Mode Teacher: Loading teacher and continuing training...')
          teach, start_epoch = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
      else:
          print('Mode Teacher: Making a teacher network from scratch and training it...')
          if args.teacher_arch != None:
            teach = load_network(True)
          else:
            teach = load_network(False)

      get_no_params(teach)
      optimizer = optim.SGD(teach.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)
      scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

      # Decay the learning rate depending on the epoch
      for e in range(0, start_epoch):
          scheduler.step()

      for epoch in tqdm(range(start_epoch, args.epochs)):
          scheduler.step()
          print('Teacher Epoch %d:' % epoch)
          if args.teacher_arch == 'darts':
              teach.drop_path_prob = args.drop_path_prob * epoch / args.epochs
          #print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
          writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)
          train_teacher(teach)
          validate(teach, args.teacher_checkpoint)

  elif args.mode == 'student':
      print('Mode Student: First, load a teacher network and convert for (optional) attention transfer')
      teach = load_network(True)

      # Very important to explicitly say we require no gradients for the teacher network
      for param in teach.parameters():
          param.requires_grad = False
      # validate(teach)
      val_losses, val_errors = [], []  # or we'd save the teacher's error as the first entry

      if args.resume:
          print('Mode Student: Loading student and continuing training...')
          student, start_epoch = load_network('checkpoints/%s.t7' % args.student_checkpoint)
      else:
          print('Mode Student: Making a student network from scratch and training it...')
          student = load_network(False)

      optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)
      scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

      # Check we aren't using KD with different output sizes

      if args.alpha > 0 and (teach.num_classes != student.num_classes):
          raise NotImplementedError('Can''t use KD with different output sizes.')

      # Decay the learning rate depending on the epoch
      for e in range(0, start_epoch):
          scheduler.step()

      for epoch in tqdm(range(start_epoch, args.epochs)):
          scheduler.step()

          if args.teacher_arch == "darts":
              teach.drop_path_prob = args.drop_path_prob * epoch / args.epochs
          if args.student_arch == "darts":
              student.drop_path_prob = args.drop_path_prob * epoch / args.epochs

          print('Student Epoch %d:' % epoch)
          print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
          writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)

          train_student(student, teach)
          validate(student, args.student_checkpoint)
