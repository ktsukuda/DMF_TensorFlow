# DMF_TensorFlow

Deep Matrix Factorization (DMF) with TensorFlow.

```
Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, Jiajun Chen
Deep Matrix Factorization Models for Recommender Systems
IJCAI 2017
```

## Environment

- Python: 3.6
- TensorFlow: 2.2.0
- CUDA: 10.1
- Ubuntu: 18.04

## Dataset

[The Movielens 1M Dataset](http://grouplens.org/datasets/movielens/1m/) is used. The rating data is included in [data/ml-1m](https://github.com/ktsukuda/DMF_TensorFlow/tree/master/data/ml-1m).

## Run the Codes

```bash
$ python DMF_TensorFlow/main.py
```

## Details

For each user, the latest and the second latest rating are used as test and validation, respectively. The remaining ratings are used as training. The hyperparameters (batch_size and lr) are tuned by using the valudation data in terms of nDCG. See [config.ini](https://github.com/ktsukuda/DMF_TensorFlow/blob/master/DMF_TensorFlow/config.ini) about the range of each hyperparameter.

By running the code, hyperparameters are automatically tuned. After the training process, the best hyperparameters and HR/nDCG computed by using the test data are displayed.

Given a specific combination of hyperparameters, the corresponding training results are saved in `data/train_result/<hyperparameter combination>` (e.g., data/train_result/batch_size_1024-lr_0.001-epoch_3-n_negative_7-top_k_10). In the directory, model files and a json file (`epoch_data.json`) that describes information for each epoch are generated. The json file can be described as follows (epoch=3).

```json
[
    {
        "epoch": 0,
        "loss": 1537454.566482544,
        "HR": 0.6372516556291391,
        "NDCG": 0.37500996278437676
    },
    {
        "epoch": 1,
        "loss": 1453089.2388916016,
        "HR": 0.6629139072847682,
        "NDCG": 0.39052323415512774
    },
    {
        "epoch": 2,
        "loss": 1425932.0260772705,
        "HR": 0.6685430463576159,
        "NDCG": 0.3928732449048126
    }
]
```
