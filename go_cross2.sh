#!/usr/bin/env bash

python main.py cifar100 student -t  wrn_40_2_cifar_10_$1 -s wrn_16_2_cifar_100_beta3_$1 --GPU $2 --alpha 0 --beta [0,0,1000] --wrn_width 2 --wrn_depth 16

python main.py cifar10 student -t  wrn_40_2_cifar_10_$1 -s wrn_16_2_cifar_10S_betaF_$1 --GPU $2 --alpha 0 --beta [1000,1000,1000] --wrn_width 2 --wrn_depth 16
python main.py cifar10 student -t  wrn_40_2_cifar_10_$1 -s wrn_16_2_cifar_10S_beta1_$1 --GPU $2 --alpha 0 --beta [1000,0,0] --wrn_width 2 --wrn_depth 16
python main.py cifar10 student -t  wrn_40_2_cifar_10_$1 -s wrn_16_2_cifar_10S_beta2_$1 --GPU $2 --alpha 0 --beta [0,1000,0] --wrn_width 2 --wrn_depth 16
python main.py cifar10 student -t  wrn_40_2_cifar_10_$1 -s wrn_16_2_cifar_10S_beta3_$1 --GPU $2 --alpha 0 --beta [0,0,1000] --wrn_width 2 --wrn_depth 16


python main.py cifar10 student -t  wrn_40_2_cifar_100_$1 -s wrn_16_2_cifar_10_beta3_$1 --GPU $2 --alpha 0 --beta [0,0,1000] --wrn_width 2 --wrn_depth 16

python main.py cifar100 student -t  wrn_40_2_cifar_100_$1 -s wrn_16_2_cifar_100S_betaF_$1 --GPU $2 --alpha 0 --beta [1000,1000,1000] --wrn_width 2 --wrn_depth 16
python main.py cifar100 student -t  wrn_40_2_cifar_100_$1 -s wrn_16_2_cifar_100S_beta1_$1 --GPU $2 --alpha 0 --beta [1000,0,0] --wrn_width 2 --wrn_depth 16
python main.py cifar100 student -t  wrn_40_2_cifar_100_$1 -s wrn_16_2_cifar_100S_beta2_$1 --GPU $2 --alpha 0 --beta [0,1000,0] --wrn_width 2 --wrn_depth 16
python main.py cifar100 student -t  wrn_40_2_cifar_100_$1 -s wrn_16_2_cifar_100S_beta3_$1 --GPU $2 --alpha 0 --beta [0,0,1000] --wrn_width 2 --wrn_depth 16
