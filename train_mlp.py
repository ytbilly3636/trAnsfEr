#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import numpy as np
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
from chainer import cuda, Variable
from set_mlp import setMLP

# preference
USE_GPU    = True
MAX_ITER   = 1000
BATCH_SIZE = 50
MLP_IN     = 3072
MLP_HIDDEN = [1000]
MLP_OUT    = 5
SUPER_CLS  = 0

if USE_GPU:
    xp = cuda.cupy
else:
    xp = np

# dataset
cifar100 = np.load('Data/cifar.npz')
train    = cifar100['train'][SUPER_CLS]
test     = cifar100['test'][SUPER_CLS]

# network
oneset = setMLP(in_size=MLP_IN, out_size=MLP_OUT, hidden=MLP_HIDDEN, active=F.relu, gpu=USE_GPU)

# train
def train_all():
    mlpgraph_x = np.arange(0, MAX_ITER)
    mlpgraph_y = np.zeros (MAX_ITER)
    aegraph_x  = np.arange(0, MAX_ITER)
    aegraph_y  = np.zeros (MAX_ITER)

    for train_iter in xrange(MAX_ITER):
        batch_index = (np.random.rand(BATCH_SIZE) * train.shape[0]).astype(np.int32)
        
        x_data_batch = []
        t_data_batch = []
        for b in batch_index:
            x_data_batch.append(train[b][0])
            t_data_batch.append(train[b][1])
            
        if not USE_GPU:
            x_batch = Variable(xp.array(x_data_batch, dtype=xp.float32))
            t_batch = Variable(xp.array(t_data_batch, dtype=np.int32))
        else:
            x_batch = Variable(cuda.to_gpu(xp.array(x_data_batch, dtype=xp.float32)))
            t_batch = Variable(cuda.to_gpu(np.array(t_data_batch, dtype=np.int32)))
        
        loss0, loss1 = oneset.train_batch(x_batch, t_batch)
        
        mlpgraph_y[train_iter] = loss0
        aegraph_y [train_iter] = loss1        
        plt.clf()
        plt.plot(mlpgraph_x, mlpgraph_y)
        plt.plot(aegraph_x, aegraph_y)
        plt.pause(0.001)
        
    oneset.save(str(SUPER_CLS))
    return
    
def test_all():
    x_data_batch = []
    t_data_batch = []
    for b in xrange(test.shape[0]):
        x_data_batch.append(train[b][0])
        t_data_batch.append(train[b][1])
        
    if not USE_GPU:
        x_batch = Variable(xp.array(x_data_batch, dtype=xp.float32))
        t_batch = Variable(xp.array(t_data_batch, dtype=np.int32))
    else:
        x_batch = Variable(cuda.to_gpu(xp.array(x_data_batch, dtype=xp.float32)))
        t_batch = Variable(cuda.to_gpu(np.array(t_data_batch, dtype=np.int32)))
        
    print '[INFO] Accuracy:', oneset.test_batch(x_batch, t_batch)
    return
    
def load(path):
    oneset.load(path)
    return
    
# ---
if __name__ == '__main__':
    #train_all()
    #load('0_2017_03_30_18:23:07')
    test_all()
    sys.exit(0)
