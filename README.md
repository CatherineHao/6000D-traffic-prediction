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

### Train & Test model:
You can use jupyter to run the Train.ipynb to train and test my models and KNN models (sorry for directly download the codes from the cloud server but not organize it into .py file). For ST-GCN, you need to run the main() to train the model and see the evaluation results. The used data is organized in './data_used/'.

### Expected Results:
Our model adds the attention mechanism to achieve better prediction results than KNN, but now it performs not as well as ST-GCN. Through analysis, we think it may be due to the sparse geographical location of the bus data used.
