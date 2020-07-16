# Text Classfication and Sentiment Analysis


## Experiments

|Model|BRIEF|
|-|-|
| CNN | Convolutional Neural Network |
| attnRNN | LSTM + Self-Attention |
| attnRNNWE | Pretrained word-embedding (PE) + LSTM + Self-Attention |
| attnBERT | BERT encoder + Self-Attention |
| attnBERTCPT | BERT encoder + Abstraction-Attention |
| attnBERTCPTDrop | BERT encoder + Abstraction-Aggregation + Aggregation Weights Dropout |
| attnTFBasic | Basic Transformer Encoder + Self-Attention |
| attnTFDefault | Transformer Encoder in torch.nn + Self-Attention |

#### IMDB (sentiment, we processed)

- Tokenized with BERT.
- Random split 8/1/1 (check if this is reasonable).
- Use Glove_42B_300D word embedding.
- Length of docs = 512
- n_class=2

| Model | Dev Accuracy | Test Accuracy | Note |
|-|-|-|-|
| CNN | 89.84 | 88.56 |
| attnRNN | 91.60 | 90.64 |
| attnRNNWE | 92.12 | 90.68 |
| attnBert | 93.52 | 92.60 |
| attnBERT + CPT(10) | 93.18 | 92.14 |
| attnBERT + CPT(20) | 93.06 | 92.34 |
| attnBERT + CPT(10) + Drop(0.01) | 93.24 | 92.22 |
| attnBERT + CPT(10) + Drop(0.02) | 93.06 | 92.14 |
| attnBERT + CPT(10) + Drop(0.05) | 92.88 | 91.82 |
| attnBERT + CPT(10) + Drop(0.10) | 92.76 | 91.50 |

##### Test New Models and Modules

| Model | Dev Accuracy | Test Accuracy | Note |
|-|-|-|-|
| attnTFBasic | 88.82 | 87.66 | Test Model. 2 Layers |
| attnTFBasic | 87.74 | 86.50 | Test Model. 6 Layers |
| attnTFDefault | 88.79 | 87.20 | Test Model. 2 Layers |
| attnTFDefault | 50.40 | 51.00 | Test Model. 6 Layers. Fail |

#### Amazon Review Beauty (sentiment)

- Tokenized with BERT.
- Random split 8/1/1.
- Use Glove_42B_300D word embedding.
- Length of docs = 200.
- resolve data imbalance problem.
- n_class=2

| Model | Dev Accuracy | Test Accuracy |
|-|-|-|
| CNN | 90.10 | 88.42 |
| attnRNN | 92.10 | 91.12 |
| attnRNNWE | 93.00 | 92.00 |
| attnBERT | 93.60 | 93.72 |
| attnBERT + CPT(10) | 93.30 | 92.52 |
| attnBERT + CPT(20) | 93.20 | 93.25 |
| attnBERT + CPT(10) + Drop(0.01) | 93.02 | 93.38 |
| attnBERT + CPT(10) + Drop(0.02) | 93.05 | 93.58 |
| attnBERT + CPT(10) + Drop(0.05) | 93.25 | 93.05 |
| attnBERT + CPT(10) + Drop(0.10) | 93.35 | 93.25 |

#### Newsroom (text categorization)

- Tokenized with BERT.
- Random split 8/1/1.
- Use Glove_42B_300D word embedding.
- Length of docs = 512.
- resolve data imbalance problem. Each category has 10000.
- n_class=5

| Model | Dev Accuracy | Test Accuracy |
|-|-|-|
| CNN | 90.68 | 90.18 |
| attnRNN | 90.32 | 89.70 |
| attnRNNWE | 91.98 | 91.26 |
| attnBERT | 92.76 | 92.28 |
| attnBERT + CPT(10) | 92.54 | 92.04 |
| attnBERT + CPT(20) | 92.34 | 92.02 |
| attnBERT + CPT(10) + Drop(0.01) | 92.86 | 92.54 |
| attnBERT + CPT(10) + Drop(0.02) | 92.62 | 92.14 |
| attnBERT + CPT(10) + Drop(0.05) | 92.78 | 92.14 |
| attnBERT + CPT(10) + Drop(0.10) | 92.60 | 92.30 |

#### AmazonReviews (text categorization)

- Tokenized with BERT.
- Random split 8/1/1.
- Use Glove_42B_300D word embedding.
- Length of docs = 200.
- resolve data imbalance problem. Each category has 10000.
- n_class=15

| Model | Dev Accuracy | Test Accuracy |
|-|-|-|
| CNN | 84.64 | 83.81 |
| attnRNN | 87.45 | 87.14 |
| attnRNNWE | 88.46 | 88.58 |
| attnBERT | 89.89 | 89.78 |
| attnBERT + CPT(10) | 89.02 | 88.95 |
| attnBERT + CPT(10) + Drop(0.01) | 88.80 | 89.00 |
| attnBERT + CPT(10) + Drop(0.02) | 88.82 | 88.87 |
| attnBERT + CPT(20) | 89.34 | 89.25 |
| attnBERT + CPT(20) + Drop(0.01) | 89.32 | 89.38 |
| attnBERT + CPT(20) + Drop(0.02) | 89.34 | 89.35 |
