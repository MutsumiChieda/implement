# 問題1
# Linear層，ReLU関数からなる3層のニューラルネットワークを定義せよ．
# ニューラルネットワークの出力は64次元のベクトルとなるようすること．

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, optimizers, serializers
from chainer import Link, Chain
from chainer.training import extensions
from chainer.backends import cuda

# Read the data
train_image = np.load('../data/train_image.npy')
train_label_raw = np.load('../data/train_label.npy')
test_image = np.load('../data/test_image.npy')
test_label_raw = np.load('../data/test_label.npy')

#------------------------------------------------
# Parameters for user

# Device to compute: 
# CPU to -1, GPU to 0
gpu_id = 0
# Labels to pass
whitelist = [0,6,9]
# Number of units for a hidden layer
mid_size = 256
# Number of output dimensions
out_size = 64
# Size of Mini batch
batchsize = 128
# Margin for computing triplet loss
triplet_margin = 0.1
# Recall to finish
recall_threshold = 0.999
#------------------------------------------------

# Encode labels to fit with the problem setting
train_label = np.array([0 if(x in whitelist) else x for x in train_label_raw])
test_label = np.array([0 if(x in whitelist) else x for x in test_label_raw])

# Data length
train_len = len(train_label)
test_len = len(test_label)
in_size = train_image.shape[1]

# Reshape data for the neural network of chainer
train_ = []
for sample in train_image:
    train_.append(np.array([sample.reshape(28,28)]))
train_ = np.array(train_)

test_ = []
for sample in test_image:
    test_.append(np.array([sample.reshape(28,28)]))
test_ = np.array(test_)

# 課題1 回答
# Define neural network consists of 
# three linear layers and reLU function
class MyNeuralNetwork(Chain):
    def __init__(self, n_mid_units, n_out=64):
        super(MyNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# Wrap the model by Classifier
model = MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size)

# Give optimizing method to the model
optimizer = optimizers.NesterovAG()
optimizer.setup(model)

# Assign GPU or CPU to the model
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
    model.to_gpu(gpu_id)
    cp = cuda.cupy

# 問題2
# Triplet lossを用いて1で定義したモデルを訓練せよ．
# その際，正常品と正常品および不良品と不良品の間の距離は小さく，
# 正常品と不良品の間の距離は大きくなるようすること．

# An Index for searching anchor image of a triplet
triplet_pos = 0
# An Index for searching positive image and negative image of a triplet
search_pos = 0

# Get one image
def get_one_image(pos):
    global search_pos
    # If index is specified, get an image there
    if(pos):
        return pos, train_[pos]

    # If index is not specified, get an image at search_pos
    data = train_[search_pos]
    idx = search_pos
    search_pos += 1
    if search_pos >= train_len:
        search_pos = 0
    return idx, data

# Get one triplet
def get_one_triplet():
    global triplet_pos
    # Get an anchor image
    anchor_label = train_label[triplet_pos]
    aidx, anchor = get_one_image(triplet_pos)
    triplet_pos += 1
    if triplet_pos >= train_len:
        triplet_pos = 0
    # Get a positive image (Its label is same as anchor)
    loop_count = 0
    pidx, positive = get_one_image(None)
    while(anchor_label != train_label[pidx] and loop_count < train_len):
        pidx, positive = get_one_image(None)
        loop_count += 1
    # Get a negative image (Its label is different from anchor)
    loop_count = 0
    nidx, negative = get_one_image(None)
    while(anchor_label == train_label[nidx] and loop_count < train_len):
        nidx, negative = get_one_image(None)
        loop_count += 1

    # # Show indices and label of the triplet
    # print("Triplet index: " + str((aidx, pidx, nidx)))
    # print("Triplet label: " + str((train_label[aidx], train_label[pidx], train_label[nidx])))

    return (anchor,positive,negative)

class TripletUpdater(training.StandardUpdater):
    def __init__(self, optimizer, device):
        self.loss_val = []
        super(TripletUpdater, self).__init__(
            None,
            optimizer,
            device=device
        )
    
    # Override the following methods 
    # so they don't raise exception when Iterator is None
    @property
    def epoch(self):
        return 0
 
    @property
    def epoch_detail(self):
        return 0.0
 
    @property
    def previous_epoch_detail(self):
        return 0.0
 
    @property
    def is_new_epoch(self):
        return False
        
    def finalize(self):
        pass
    
    def update_core(self):
        # Get Optimizer
        optimizer = self.get_optimizer('main')
        
        # Get triplet
        anchor = []
        positive = []
        negative = []
        for i in range(batchsize):
            in_data = get_one_triplet()
            anchor.append(in_data[0])
            positive.append(in_data[1])
            negative.append(in_data[2])

        if(gpu_id >= 0):
            # Use array of cupy to use GPU for training
            anchor = cp.array(anchor)
            positive = cp.array(positive)
            negative = cp.array(negative)
        else:
            # Use array of numpy to use CPU for training
            anchor = np.array(anchor)
            positive = np.array(positive)
            negative = np.array(negative)

        model = optimizer.target
        
        # Update triplet by neural network
        anchor_r = model(anchor)
        positive_r = model(positive)
        negative_r = model(negative)

        # Learn w/ Triplet loss function
        optimizer.update(F.triplet, anchor_r, positive_r, negative_r, margin=triplet_margin)

# Get an updater that uses the Iterator and Optimizer
updater = TripletUpdater(optimizer, device=gpu_id)

# Setup a Trainer
trainer = training.Trainer(updater, (2000, 'iteration'), out="result")

# Add some extensions for reporting
trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.extend(extensions.LogReport())

# Run
trainer.run()

# Save trained model
serializers.save_npz('my.model', model)

# 課題3
# 訓練されたモデルを使い，評価セット中のそれぞれのサンプルについて不良度を推定せよ．
# 学習データ中の正常品を用いて基準となるベクトルを作成し，その基準ベクトルからの距離を不良度とすること．
# model = MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size)
# serializers.load_npz('my.model', model)

# Get samples of qualified product
qualified_samples = []
for i in range(train_len):
    if(train_label[i] in whitelist):
        qualified_samples.append(train_[i])
qualified_samples = cp.array(qualified_samples) if(gpu_id >= 0) else np.array(qualified_samples)

# Compute base vector using qualified product
result = model(qualified_samples).data
base_vector = np.mean(result, axis=0)

# Compute badness from base vector
badness = []
if(gpu_id >= 0):
    test_ = cp.array(test_)
result = model(test_).data
if(gpu_id >= 0):
    # Memo: Convert result vector of test sample and base vector before computing
    # to prevent linalg.norm method from returning cupy's ndarray when it takes cupy's ndarray
    result = cp.asnumpy(result)
    base_vector = cp.asnumpy(base_vector)
badness = [np.linalg.norm(result[idx] - base_vector) for idx in range(test_len)]

# Visualize the distribution of badness
import matplotlib.pyplot as plt
fig = plt.figure()
fig.clf()
plt.grid()

ax = fig.add_subplot(1,1,1)
ax.hist(badness, bins=100)
ax.set_title('Badness histogram')
ax.set_xlabel('badness')
ax.set_ylabel('freq')
plt.savefig("badness_histogram.png")

# 問題4
# モデルを評価せよ．評価指標としては，Recall ≥ 0.999を満たす検出閾値を
# 設定した場合の True Negative Rate（真陰性率）を用いること．

print("threshold recall    selectivity ")
test_label = np.array([0 if(test_label_ in whitelist) else 1 for test_label_ in test_label])
badness_rng = np.sort(badness)
badness_rng = (badness_rng[:-1] + badness_rng[1:]) / 2.0
selectivity_values = []
result = []

# Increase threshold until recall excesses "recall_threshold"
for threshold in badness_rng:
    # Label the test samples whether qualified or not
    preds = np.array([0 if(x < threshold) else 1 for x in badness])
    
    # Compute recall
    true_positive, false_negative, true_negative, false_positive = [0,0,0,0]
    for idx in range(test_len):
        actual, pred = test_label[idx], preds[idx]
        if(actual):
            if(pred):
                true_positive += 1
            else:
                false_negative += 1
        else:
            if(pred):
                false_positive += 1
            else:
                true_negative += 1
    recall = true_positive / (true_positive + false_negative)
    selectivity = true_negative / (true_negative + false_positive)

    print("%9.7f " % threshold, end='')
    print("%9.7f " % recall, end='')
    print("%9.7f " % selectivity, end='\r')
    
    result.append([recall, selectivity])
    if(recall > recall_threshold):
        selectivity_values.append([threshold,selectivity])
    else:
        # break
        pass
result = np.array(result)
result_len = len(result)
selectivity_values = np.array(selectivity_values)

# Export the prediction
threshold = selectivity_values[np.argmax(selectivity_values[:,1]), 0]
preds = np.array([0 if(x < threshold) else 1 for x in badness])
np.save('../output/preds.npy', preds)
np.savetxt('../output/preds.csv', preds, delimiter=',')

# Show result w/ graph
import matplotlib.pyplot as plt
plt.clf()
plt.title('Recall and selectivity over threshold')
plt.plot(badness_rng[:result_len], result[:,0], 'b', label='recall')
plt.plot(badness_rng[:result_len], result[:,1], 'g', label='selectivity')
plt.legend()
plt.grid()
plt.savefig('best_threshold.png')
print("\n\n")
print("Best threshold  : %6.5f" % threshold)
print("Max selectivitiy: %6.5f" % max(selectivity_values[:,1]))
