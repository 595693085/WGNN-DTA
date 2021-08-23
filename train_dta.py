import sys
import torch
import torch.nn as nn
# sys.path.append('./')
from dta_components.data_process import create_DTA_dataset
from dta_components.utils import *
from metric import *
from model import DTA_GCN, DTA_GAT
import os

DTAdatasets = [['davis', 'kiba'][int(sys.argv[1])]]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000
if_valid = False

model_type = DTA_GCN

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(results_dir, DTAdatasets[0])):
    os.makedirs(os.path.join(results_dir, DTAdatasets[0]))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = model_type()
model.to(device)
model_st = model_type.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

bset_val = 10000
for dataset in DTAdatasets:
    # train_data, valid_data, test_data = create_DTA_dataset(dataset)
    train_data, test_data = create_DTA_dataset(dataset)
    valid_loader = None
    if if_valid:
        train_l = int(0.8 * len(train_data))
        valid_l = len(train_data) - int(0.8 * len(train_data))
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_l, valid_l])
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                   collate_fn=collate)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    model_file_name = 'models/model_' + model_st + '_' + dataset + '.model'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)
        print('predicting for test data')
        T, P = predicting(model, device, test_loader)
        val = get_mse(T, P)
        print('test result:', val)
        if if_valid:
            T_V, P_v = predicting(model, device, valid_loader)
            val_V = get_mse(T_V, P_v)
            if val_V < bset_val:
                print('model saved and valid mse:', val_V)
                torch.save(model.state_dict(), model_file_name)
    if not if_valid and NUM_EPOCHS != 0:
        torch.save(model.state_dict(), model_file_name)
    # test
    print('all training done. Testing...')
    model_p = model_type()
    model_p.to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_T, test_P = predicting(model_p, device, test_loader)
    test_mse = get_mse(test_T, test_P)
    test_PI = get_ci(test_T, test_P)
    test_pearson = get_pearson(test_T, test_P)
    result_str = 'test result:' + '\n' + 'test_mse:' + str(test_mse) + '\n' + 'test_PI:' + str(
        test_PI) + '\n' + 'test_pearson:' + str(test_pearson) + '\n'

    print(result_str)
    save_file = os.path.join(results_dir, dataset, 'test_restult_' + model_st + '.txt')
    open(save_file, 'w').writelines(result_str)
