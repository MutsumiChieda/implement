# MNIST 不良品識別

Those script are done in ipython 7.3.0

## まずは3層NNを作る(活性化はReLU関数)

Trainerを使えば、後が楽になりそう  
[MNIST using Trainer](https://docs.chainer.org/en/stable/examples/mnist.html)

## TrainとTestのラベルを調べる

```python
unique, counts = np.unique(train_label, return_counts=True)
dict(zip(unique, counts))
```

> {0: 5823, 1: 60, 4: 44, 5: 53, 6: 5918, 8: 54, 9: 5949}

```python
unique, counts = np.unique(test_label, return_counts=True)
dict(zip(unique, counts))
```

> {0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}

TrainとTestで値の分布が全然違うから、学習がうまくいかない
(精度 0.5132)

## Triplet lossを使う

ΣiNmax[d(xai,xpi)−d(xai,xni)+α,0]

ラベルが同じ（プラス）同士のペアよりも、ラベルが違う（マイナス）同士のほうが近くにあったら、ペナルティをつけなさい。つまり、プラスのペアよりも、マイナスのペアのほうが遠くにあるようにプロットしなさい

<!-- [例外画像の検出ニューラルネットワーク](https://cocon-corporation.com/cocontoco/find_anomaly_values/) -->

[Chainer v4 ビギナー向けチュートリアル](https://qiita.com/mitmul/items/1e35fba085eb07a92560)
