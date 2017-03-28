#!/usr/bin/python
#-*- coding:utf-8 -*-

import os, datetime
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, serializers, Chain

class MLP(Chain):
    def __init__(self, in_size, out_size, hidden, active=None):
        # in_size:  number of units in input layer
        # out_size: number of units in output layer
        # hidden:   [0] number of units in hidden layers
        #           [1] number of hidden layers (not used)
        # active:   activation function (use chainer.functions)
        super(MLP, self).__init__(
            l_in  = L.Linear(in_size, hidden[0]),
            l_hid = L.Linear(hidden[0], hidden[0]),
            l_out = L.Linear(hidden[0], out_size),
        )
        self.act = active
        return
        
    def __call__(self, x):
        # x:    Variable as input
        if self.act == None:
            h = self.l_in(x)
            h = self.l_hid(h)
            return self.l_out(h)
        else:
            h = self.act(self.l_in(x))
            h = self.act(self.l_hid(h))
            return self.l_out(h)
            
    def predict(self, x):
        y = self.__call__(x)
        return np.argmax(y.data)
        
class AE(Chain):
    def __init__(self, in_size, hid_size, active=None):
        # in_size:  number of units in input layer
        # hid_size: number of hidden layers
        # active:   activation function (use chainer.functions)
        super(AE, self).__init__(
            l_in  = L.Linear(in_size, hid_size),
            l_out = L.Linear(hid_size, in_size),
        )
        self.act = active
        
    def __call__(self, x):
        # x:    Variable as input
        if self.act == None:
            h = self.l_in(x)
            return self.l_out(h)
        else:
            h = self.act(self.l_in(x))
            return self.l_out(h)
            
    def loss(self, x):
        return F.mean_squared_error(self.__call__(x), x)

class setMLP():
    def __init__(self, in_size, out_size, hidden, active=None, gpu=False):
        # in_size:  number of units in input layer
        # out_size: number of units in output layer
        # hidden:   [0] number of units in hidden layers
        #           [1] number of hidden layers
        # active:   activation function (use chainer.functions)
        self.mlp = MLP(in_size, out_size, hidden, active)
        self.ae  = AE(in_size, hidden[0], active)
        
        self.cls_mlp = L.Classifier(self.mlp)
        self.opt_mlp = optimizers.Adam()
        self.opt_ae  = optimizers.Adam()
        self.opt_mlp.setup(self.cls_mlp)
        self.opt_ae.setup(self.ae)
        
        if gpu:
            self.cls_mlp.to_gpu()
            self.ae.to_gpu()
            
        print '[INFO] Done construct network'
        return
        
    def train(self, batch_x, batch_t):
        # batch_x:  Variable type
        # batch_t:  Variable type
      
        # train mlp
        self.opt_mlp.update(self.cls_mlp, batch_x, batch_t)
        
        # train ae
        self.ae.zerograds()
        ae_loss = self.ae.loss(batch_x)
        ae_loss.backward()
        self.opt_ae.update()
        
        # [0]:  loss of mlp
        # [1]:  loss of ae
        return self.cls_mlp.loss.data, ae_loss.data
    
    def save(self, path=None):
        d = datetime.datetime.today()
        serializers.save_npz(d.strftime("%Y_%m_%d_%H:%M:%S_mlp.model"), self.mlp)
        serializers.save_npz(d.strftime("%Y_%m_%d_%H:%M:%S_ae.model"),  self.ae)
        
        print '[INFO] Done saving models'
        return
        
    def load(self, path):
        serializers.load_npz(path + '_mlp.model', self.mlp)
        serializers.load_npz(path + '_ae.model' , self.ae)    
        
        print '[INFO] Done Loading models'
        return
        
    def predict(self, x):
        return self.mlp.predict(x)
        
    def similarity(self, batch_x):
        batch_y = self.ae(batch_x)
        return F.mean_squared_error(batch_x, batch_y).data
        

# test script        
if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from chainer import datasets
    
    train, test = datasets.get_mnist()
    
    mset = setMLP(
        in_size  = 784,
        out_size = 10,
        hidden   = [100, 1],
        active   = F.relu,
        gpu      = False,
    )
    
    MAX_ITER = 100
    
    mlpgraph_x = np.arange(0, MAX_ITER)
    mlpgraph_y = np.zeros(MAX_ITER)
    aegraph_x = np.arange(0, MAX_ITER)
    aegraph_y = np.zeros(MAX_ITER)
    
    for train_iter in xrange(MAX_ITER):
        batch_data_x = []
        batch_data_t = []
        for batch in xrange(100):
            batch_data_x.append(train[batch + train_iter * 100][0])
            batch_data_t.append(train[batch + train_iter * 100][1])
        batch_x = Variable(np.array(batch_data_x, dtype=np.float32))
        batch_t = Variable(np.array(batch_data_t, dtype=np.int32))
        
        loss0, loss1 = mset.train(batch_x, batch_t)
        
        mlpgraph_y[train_iter] = loss0
        aegraph_y[train_iter] = loss1        
        plt.clf()
        plt.plot(mlpgraph_x, mlpgraph_y)
        plt.plot(aegraph_x, aegraph_y)
        plt.pause(0.001)
        
    mset.save()
    sys.exit(0)
