import os
import sys

import numpy as np
import torch
import torch.nn as nn
# sys.path.append('./')
from dta_components.data_process import create_CPI_dataset
from dta_components.utils import *
from metric import *
from model import CPI_GCN, CPI_GAT
from sklearn.metrics import roc_auc_score, precision_score, recall_score

CPIdatasets = [['human', 'celegans'][int(sys.argv[1])]]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
ratio = [1, 3, 5][int(sys.argv[3])]
print('cuda_name:', cuda_name)
print('dataset:', CPIdatasets[0])
print('ratio', ratio)

model_type = CPI_GCN
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.001
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(results_dir, CPIdatasets[0])):
    os.makedirs(os.path.join(results_dir, CPIdatasets[0]))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = model_type()
model.to(device)
model_st = model_type.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for dataset in CPIdatasets:
    # train_data, valid_data, test_data = create_DTA_dataset(dataset)
    train_data, dev_data, test_data = create_CPI_dataset(dataset, ratio)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(ratio) + '.model'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)
        print('predicting for test data')
        T_D, P_D = predicting(model, device, dev_loader)
        T, P = predicting(model, device, test_loader)
        val_auc_score = roc_auc_score(T, P)
        dev_auc_score = roc_auc_score(T_D, P_D)
        print('test result:', val_auc_score, dev_auc_score)

    if NUM_EPOCHS != 0:
        torch.save(model.state_dict(), model_file_name)
    # test
    print('all training done. Testing...')
    model_p = model_type()
    model_p.to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_T, test_P = predicting(model_p, device, test_loader)
    test_auc = roc_auc_score(test_T, test_P)
    test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_f1_score=f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
    result_str = 'test result:' + '\n' + 'test_auc:' + str(test_auc) + '\n' + 'test_recall:' + str(
        test_recall) + '\n' + 'test_precision:' + str(test_precision) + '\n'+ 'test_f1_core:' + str(test_f1_score) + '\n'

    print(result_str)

    save_file = os.path.join(results_dir, dataset, 'test_restult_' + str(ratio) + '_' + model_st + '.txt')
    open(save_file, 'w').writelines(result_str)
