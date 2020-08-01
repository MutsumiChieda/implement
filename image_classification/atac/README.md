# [Attention is Activation](https://arxiv.org/pdf/2007.07729.pdf)
ReLUに注意機構とバッチ正規化を入れると少ない計算量で表現力が増すことを示した論文

## Usage
```shell
cd script
python main.py
```
Performance result will be generated in `plot/`

## Arguments
|Parameter|Description|Expected|
|---------|-----------|--------|
|-a|Use ATAC|-|
|-e|# of epoch|int|
|-r|Reduction rate of channels in ATAC unit|float(0~1)|