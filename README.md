# ACCE
Automatic Corpus-level and Concept-based Explanation for Text Classfication Models.

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/tshi04/DMSC_FEDA/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/DMSC_FEDA/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/DMSC_FEDA/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/pdf/2009.09112.pdf)

This repository is a pytorch implementation for the following arxiv paper:

#### [A Concept-based Abstraction-Aggregation Deep Neural Network for Interpretable Document Classification](https://arxiv.org/pdf/2004.13003.pdf)
[Tian Shi](http://people.cs.vt.edu/tshi/homepage/home), 
[Xuchao Zhang](https://xuczhang.github.io/), 
[Ping Wang](http://people.cs.vt.edu/ping/homepage/), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/)

## Requirements

- Python 3.6.9
- argparse=1.1
- torch=1.4.0
- sklearn=0.22.2.post1
- numpy=1.18.2

## Dataset

Please download processed dataset from here. Place them along side with DMSC_FEDA.

```bash
|--- ACCE
|--- Data
|    |--- imdb_data
|    |--- newsroom_data
|    |    |--- dev
|    |    |--- glove_42B_300d.npy
|    |    |--- test
|    |    |--- train
|    |    |--- vocab
|    |    |--- vocab_glove_42B_300d
|--- nats_results (results, automatically build)
|
```

## Usuage

```Training, Validate, Testing``` python3 run.py --task train
```Testing only``` python3 run.py --task test
```Evaluation``` python3 run.py --task evaluate
```keywords Extraction``` python3 run.py --task keywords_attnself
```keywords Extraction``` python3 run.py --task keywords_attn_abstraction
```Attention Weight Visualization``` python3 run.py --task visualization

If you want to run baselines, you may need un-comment the corresponding line in ```run.py```.

## Baselines Implemented.

| Model | BRIEF | 
| ------ | ------ |
| CNN | Convolutional Neural Network |
| RNNAttn | Bi-LSTM + Self-Attention |
| RNNAttnWE | RNNAttn + Pretrained Word Embedding |
| RNNAttnWECPT | RNNAttnWE + Concept Based |
| RNNAttnWECPTDrop | RNNAttnWECPT + Attention Weights Dropout |
| Bert* | Replace RNN with BERT |

# Use Pretrained Model

Coming Soon.

## Citation

```
@article{shi2020concept,
  title={A Concept-based Abstraction-Aggregation Deep Neural Network for Interpretable Document Classification},
  author={Shi, Tian and Zhang, Xuchao and Wang, Ping and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2004.13003},
  year={2020}
}
```
