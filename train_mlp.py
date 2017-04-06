#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys, datetime
import numpy as np
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
from chainer import cuda, Variable
from set_mlp import setMLP

# preference
USE_GPU    = True
MAX_ITER   = 5000
BATCH_SIZE = 50
SEARCH_SIZE= 500
MLP_IN     = 3072
MLP_HIDDEN = [1000]
MLP_OUT    = 5
SUPER_CLS  = 10
DATA_BASE  = [
    'Result/init_rand/0_2017_04_03_14:59:32',
    'Result/init_rand/1_2017_04_03_15:09:48',
    'Result/init_rand/2_2017_04_03_15:20:01',
    'Result/init_rand/3_2017_04_03_15:29:36',
    'Result/init_rand/4_2017_04_03_15:43:09',
    'Result/init_rand/5_2017_04_03_15:55:06',
    'Result/init_rand/6_2017_04_03_17:05:50',
    'Result/init_rand/7_2017_04_03_17:26:41',
    'Result/init_rand/8_2017_04_03_17:37:45',
    'Result/init_rand/9_2017_04_03_17:47:26'
]

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
            t_batch = Variable(cuda.to_gpu(xp.array(t_data_batch, dtype=np.int32)))
        
        loss0, loss1 = oneset.train_batch(x_batch, t_batch)
        
        mlpgraph_y[train_iter] = loss0
        aegraph_y [train_iter] = loss1        
        plt.clf()
        plt.ylim(0, 5)
        plt.plot(mlpgraph_x, mlpgraph_y)
        plt.plot(aegraph_x, aegraph_y)
        plt.pause(0.001)
        
        if train_iter % 100 == 0:
            print '[INFO] Varidation: ', train_iter
            test_all()
    
    d = datetime.datetime.today()
    oneset.save('Result/' + str(SUPER_CLS) + d.strftime('_%Y_%m_%d_%H:%M:%S'))
    plt.savefig('Result/' + str(SUPER_CLS) + d.strftime('_%Y_%m_%d_%H:%M:%S') + '.png')
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
    
def search_db():
    batch_index = (np.random.rand(SEARCH_SIZE) * train.shape[0]).astype(np.int32)
        
    x_data_batch = []
    for b in batch_index:
        x_data_batch.append(train[b][0])        
    if not USE_GPU:
        x_batch = Variable(xp.array(x_data_batch, dtype=xp.float32))
    else:
        x_batch = Variable(cuda.to_gpu(xp.array(x_data_batch, dtype=xp.float32)))

    distance = []
    for db in DATA_BASE:
        oneset.load(db)
        dist = oneset.similarity(x_batch)
        distance.append(dist)
    
    nearest = np.argmin(distance)
    print '[INFO] Dataset No.', nearest, 'is suitable'
    return nearest
    
def load(path):
    oneset.load(path)
    return
    
# ---
if __name__ == '__main__':
    # let the network learn images in the SUPER_CLS
    #train_all()
    
    # search the nearest dataset w.r.t SUPER_CLS
    # after that load the nearest one & let the network learn
    load(DATA_BASE[search_db()])
    train_all()
    
    sys.exit(0)
