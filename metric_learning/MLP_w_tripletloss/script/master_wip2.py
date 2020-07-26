import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers, serializers, reporter
from chainer import Link, Chain
from chainer.training import extensions
from chainer.backends import cuda
cp = cuda.cupy

# Read data
train_images = np.load('../data/train_image.npy')
train_label_raw = np.load('../data/train_label.npy')
test_images = np.load('../data/test_image.npy')
test_label_raw = np.load('../data/test_label.npy')

# Fix random seeds for reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    cp.random.seed(seed)
set_random_seed(0)

#------------------------------------------------
# Parameters for user

# Device to compute: 
# CPU to -1, GPU to 0
gpu_id = -1
# Labels to pass
whitelist = [0,6,9]
# Number of units for a hidden layer
mid_size = 256
# Number of output dimensions
out_size = 64
# Number of epoch
n_epoch = 50
# Size of Mini batch
batchsize = 100
# How many times to train the same sample
repeat = 1
# Optimizer algorithm 
optimizer = optimizers.CorrectedMomentumSGD()
# Margin for computing triplet loss
triplet_margin = 0.3
# Recall to finish
recall_threshold = 0.999
#------------------------------------------------

# Encode labels to fit with the problem setting
train_label = np.array([0 if(x in whitelist) else x for x in train_label_raw])
test_label = np.array([0 if(x in whitelist) else x for x in test_label_raw])

# Data length
train_len = len(train_label)
test_len = len(test_label)
in_size = train_images.shape[1]

# 課題1 回答
# Define neural network consists of 
# three linear layers and reLU function
class MyNeuralNetwork(Chain):
    def __init__(self, n_mid_units=256, n_out=64):
        super(MyNeuralNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# 問題2
# Triplet lossを用いて1で定義したモデルを訓練せよ．
# その際，正常品と正常品および不良品と不良品の間の距離は小さく，
# 正常品と不良品の間の距離は大きくなるようすること．

# Get one image
triplet = []
search_pos = 0
def get_one_image(images):
    global search_pos

    # Get an image at search_pos
    image = images[search_pos]
    idx = search_pos
    search_pos += 1
    if(search_pos >= len(images)):
        search_pos = 0
    return idx, image

# Get triplets of each sample
def get_triplet(images,label):
    data_len = len(label)
    triplet = []
    for idx in range(data_len):
        # Get an anchor image
        anchor_label = label[idx]
        aidx, anchor = idx, images[idx]
        
        # Get a positive image (Its label is same as anchor)
        loop_count = 0
        pidx, positive = get_one_image(images)
        while(anchor_label != label[pidx] and loop_count < data_len):
            pidx, positive = get_one_image(images)
            loop_count += 1
        
        # Get a negative image (Its label is different from anchor)
        loop_count = 0
        nidx, negative = get_one_image(images)
        while(anchor_label == label[nidx] and loop_count < data_len):
            nidx, negative = get_one_image(images)
            loop_count += 1

        triplet.append(np.array([anchor, positive, negative]).flatten())

        # # Show indices and label of the triplet
        # print("Triplet index: " + str((aidx, pidx, nidx)))
        # print("Triplet label: " + str((label[aidx], label[pidx], label[nidx])))
    
    return cp.array(triplet) if(gpu_id >= 0) else np.array(triplet)

train_triplet = get_triplet(train_images, train_label)
test_triplet = get_triplet(test_images, test_label)

class TripletClassifier(Chain):
    def __init__(self, predictor):
        super(TripletClassifier, self).__init__(predictor=predictor)

    def __call__(self, x, y):
        anchor, positive, negative = (x[:,:in_size], x[:,in_size:(in_size*2)], x[:,(in_size*2):])
        anchor_ = self.predictor(anchor)
        positive_ = self.predictor(positive)
        negative_ = self.predictor(negative)
        loss = F.triplet(anchor_, positive_, negative_, margin=triplet_margin)
        reporter.report({'loss': loss}, self)
        return loss
    
    def predict(self, x):
        return self.predictor(x)

model = TripletClassifier(MyNeuralNetwork(n_mid_units=mid_size, n_out=out_size))

# Assign GPU or CPU to the model
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
    model.to_gpu(gpu_id)

# Define Iterator
train_set = datasets.TupleDataset(train_triplet, train_label)
test_set = datasets.TupleDataset(test_triplet, test_label)
train_iter = iterators.SerialIterator(train_set, batchsize)
test_iter = iterators.SerialIterator(test_set, batchsize, repeat=False, shuffle=False)

# Define optimizers
optimizer.setup(model)

# Give the iterators and optimizers to updater
updater = training.StandardUpdater(train_iter, optimizer)

# Give trigger for early stopping 
stop_trigger = training.triggers.EarlyStoppingTrigger(
    monitor='validation/main/loss',
    max_trigger=(n_epoch, 'epoch'))

# Give updater to trainer
trainer = training.Trainer(updater, stop_trigger)

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='loss.png'))

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

# Train the model
trainer.run()

# Save trained model
serializers.save_npz('my.model', model)

# 課題3
# 訓練されたモデルを使い，評価セット中のそれぞれのサンプルについて不良度を推定せよ．
# 学習データ中の正常品を用いて基準となるベクトルを作成し，その基準ベクトルからの距離を不良度とすること．

# Get samples of qualified product
qualified_samples = []
for i in range(train_len):
    if(train_label[i] in whitelist):
        qualified_samples.append(train_images[i])
qualified_samples = cp.array(qualified_samples) if(gpu_id >= 0) else np.array(qualified_samples)

# Compute base vector using qualified product
result = model.predict(qualified_samples).data
base_vector = cp.mean(result, axis=0) if(gpu_id >= 0) else np.mean(result, axis=0)

# Compute badness from base vector
badness = []
test_samples = cp.array(test_images) if(gpu_id >= 0) else np.array(test_images)
result = model.predict(test_samples).data
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
        break
        # pass
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
plt.xlabel('detection threshold')
plt.ylabel('selectivity value')
plt.legend()
plt.grid()
plt.savefig('tnrate.png')
print("\n\n")
print("Best threshold  : %6.5f" % threshold)
print("Max selectivitiy: %6.5f" % max(selectivity_values[:,1]))