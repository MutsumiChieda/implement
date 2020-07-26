# -*- coding: utf-8 -*-

# 問題1
# Linear層，ReLU関数からなる3層のニューラルネットワークを定義せよ．
# ニューラルネットワークの出力は64次元のベクトルとなるようすること．

import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist

# Read data
train_image = np.load('../data/train_image.npy')
train_label = np.load('../data/train_label.npy')
test_image = np.load('../data/test_image.npy')
test_label = np.load('../data/test_label.npy')

# In this problem '0', '6', '9' are Normal
# and others are Defective
def encoder(x):
    return 0 if(x in [0,6,9]) else 1

# Reshape data
train = []
for i in range(len(train_label)):
    train.append((train_image[i], encoder(train_label[i])))
test = []
for i in range(len(test_label)):
    test.append((test_image[i], encoder(test_label[i])))

# Parameters to create neural network
in_size = train_image.shape[1]
out_size = 64
mid_size = int(np.sqrt(in_size*out_size))

# Define Iterator
batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# 課題1 回答
# Define neural network consists of linear layers and reLU function
class MyNeuralNetwork(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# Assign GPU
gpu_id = 0
model = MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size)
if gpu_id >= 0:
    model.to_gpu(gpu_id)

max_epoch = 20

# Wrap the model by Classifier
model = L.Classifier(model)

# Select optimizing method
optimizer = optimizers.MomentumSGD()

# Give the optimizer a reference to the model
optimizer.setup(model)

# Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Setup a Trainer
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

# Add some extensions for reporting
from chainer.training import extensions
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

# Run
trainer.run()

import matplotlib.pyplot as plt

model = MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size)
serializers.load_npz('mnist_result/model_epoch-10', model)

# Show the output
# and evaluate its recall
true_positive = 0
false_negative = 0
preds = []
for sample in test:
    x, t = sample
    y = model(x[None, ...])
    pred = y.array.argmax(axis=1)[0]
    preds.append(pred)
    # print('\nlabel:', t, end='')
    # print('   pred:', pred, end='')
    # if(t != pred):
    #     print('   Incorrect!',end='')
    if(t==1):
        if(pred==1):
            true_positive += 1
        else:
            false_negative += 1
recall = true_positive / (true_positive + false_negative)
print('\nrecall :' + str(recall))