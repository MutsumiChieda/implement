# 問題1
# Linear層，ReLU関数からなる3層のニューラルネットワークを定義せよ．
# ニューラルネットワークの出力は64次元のベクトルとなるようすること．

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import iterators, optimizers, serializers
from chainer import Link, Chain
from chainer import training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import matplotlib.pyplot as plt

# Read data
train_image = np.load('../data/train_image.npy')
train_label = np.load('../data/train_label.npy')
test_image = np.load('../data/test_image.npy')
test_label = np.load('../data/test_label.npy')

# Parameters
max_epoch = 10

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

valid_num=(int(len(train)*0.3))
valid, train = split_dataset_random(train, valid_num, seed=0)

# Define Iterator
batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# 課題1 回答
# Define neural network consists of linear layers and reLU function
class MyNeuralNetwork(Chain):
    def __init__(self, n_mid_units=128, n_out=64):
        super(MyNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# Create model
in_size = train_image.shape[1]
out_size = 64
mid_size = int(np.sqrt(in_size*out_size))
model = MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size)

# Assign GPU
gpu_id = 0
if gpu_id >= 0:
    model.to_gpu(gpu_id)

# Wrap w/ Classifier and include loss function and etc. into a model
# model = L.Classifier(model)
class TripletLossClassifier(L.Classifier, lossfun=F.triplet):
    # Override forward method 
    # so triplet function can take 3 variables 
    # anchor, positive and negative
    def forward(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(*args, **kwargs)
        print(self.y, t)
        # # Get positive and negative sample for triplet
        # # positive: same class sample
        # # negative: different class sample
        # positive = train
        # negative = train
        # self.loss = self.lossfun(self.y, positive, negative)

        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

# model = L.Classifier(model)
model = TripletLossClassifier(model, lossfun=F.triplet)
optimizer = optimizers.SGD(lr=0.01).setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
trainer = training.Trainer(
    updater, (max_epoch, 'epoch'), out='mnist_result')

# Extensions for trainer
trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.ParameterStatistics(model.predictor.l1, {'std': np.std}))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

# Run 
trainer.run()

# Evaluate w/ test data
test_evaluator = extensions.Evaluator(test_iter, model, device=gpu_id)
results = test_evaluator()
print('Test accuracy:', results['main/accuracy'])