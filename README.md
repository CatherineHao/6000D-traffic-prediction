# 6000D-traffic-prediction
The final project for 6000D: GCN for traffic forecast.

The code is based on: https://github.com/hazdzz/STGCN

#### Existing Problems: Now this work is incomplete. Due to the sparse data of buses, the model results are not good and need to be modified. 

### Code Overview:
```
data_used\
        knn_dataset (the train dataset processed for KNN prediction model)
        metr-la (the metr-la dataset)
        prediction_dataset (the used data in our model after data processing, including train, test and validation datasets)
        data_toyset.csv (the Shenzhen bus data after rough processing)
data_prepare.ipynb (the data processing code. The intermediate result of data can be seen here)
STGCN_test.py (the main code of ST-GCN) 
Train.ipynb (train and test KNN; train and test our model)
```

### Requirement:
```
numpy~=1.22.4
pandas~=1.5.2
scikit_learn~=1.1.2
conda~=4.10.3
scipy~=1.9.3
torch~=1.11.0+cu113
tqdm~=4.61.2
```
