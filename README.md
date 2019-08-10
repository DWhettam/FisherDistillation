# Network Distillation using a Fisher expanded teacher


## Installation Instructions

If installing with conda:

```
conda create -n torch python=3.6
source activate torch
```
then

```
conda install pytorch torchvision -c pytorch
pip install tqdm
pip install tensorboardX
conda install tensorflow
```

## Training a Teacher

In general, the following code trains a teacher network:

```
python main.py <DATASET> teacher -t <TEACHER_CHECKPOINT> --teach_depth <TEACHER_DEPTH> --teach_width <TEACHER_WIDTH>
```

Where `<DATASET>` is one of `cifar10`, `cifar100` or `imagenet`. By
default, `cifar10` and `cifar100` are assumed to be stored at
`/disk/scratch/datasets/cifar`, but any directory can be set with
`--cifar_loc`.

In the paper, results are typically reported using a standard 40-2 WRN,
which would be the following (on cifar-10):

```
python main.py cifar10 teacher --conv Conv -t wrn_40_2 --wrn_depth 40 --wrn_width 2
```

## Training a Student

Students can be trained using KD (by setting alpha>0) and/or AT (by setting beta>0) as:

```
python main.py <DATASET> student -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --student_depth <STUDENT_DEPTH> --student_width <STUDENT_WIDTH> --alpha <ALPHA for KD> --beta <BETA for AT>
```
  
Note: the AT method uses KD by default, so to turn it off, set alpha to 0

As an example, this would train a model with the same structure as the
teacher network, but using a bottleneck grouped + pointwise convolution as
a substitute for the full convolutions in the full network with attention transfer:

```
python main.py cifar10 student --conv G8B2 -t wrn_40_2 -s wrn_40_2.g8b2.student --wrn_depth 40 --wrn_width 2 --alpha 0. --beta 1e3
```

## Acknowledgements

The following repos provided basis and inspiration for this work:

```
https://github.com/szagoruyko/attention-transfer
https://github.com/kuangliu/pytorch-cifar
https://github.com/xternalz/WideResNet-pytorch
https://github.com/ShichenLiu/CondenseNet
```

## Citing this work

If you would like to cite this work, please use the following bibtex entry:

