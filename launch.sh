#!/bin/sh

python3 train.py --net wdeeplabv3p \
                 --wn haar \
                 --backbone resnet101 \
                 --dataset pascal \
                 --epochs 1000 \
                 --no_cuda \
                 --workers 0