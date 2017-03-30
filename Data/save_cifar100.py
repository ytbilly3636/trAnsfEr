#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import chainer
from   chainer import datasets

def ret_index(val, lis):
    for i, lab in enumerate(lis):
        if val in lab:
            return i, lab.index(val)

print '[INFO] Fetch cifar100 dataset ...'
train, test = datasets.get_cifar100(ndim=1)
print '[INFO] Done fetch!'

new_train = [[] for i in xrange(20)]
new_test  = [[] for i in xrange(20)]

labels = [
    [ 4, 30, 55, 72, 95],
    [ 1, 32, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [ 9, 10, 16, 28, 61],
    [ 0, 51, 53, 57, 83],
    [22, 39, 40, 86, 87],
    [ 5, 20, 25, 84, 94],
    [ 6,  7, 14, 18, 24],
    [ 3, 42, 43, 88, 97],
    [12, 17, 37, 68, 76],
    [23, 33, 49, 60, 71],
    [15, 19, 21, 31, 38],
    [34, 63, 64, 66, 75],
    [26, 45, 77, 79, 99],
    [ 2, 11, 35, 46, 98],
    [27, 29, 44, 78, 93],
    [36, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [ 8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89]
]

print '[INFO] Remaking dataset ...'
for t in train:
    super_class, sub_class = ret_index(t[1], labels)
    new_train[super_class].append([t[0], sub_class])

for t in test:
    super_class, sub_class = ret_index(t[1], labels)
    new_test[super_class].append([t[0], sub_class])
print '[INFO] Done remaking!'

print '[INFO] Save ...'
np.savez('cifar.npz', train=new_train, test=new_test)
print '[INFO] Done save!'
