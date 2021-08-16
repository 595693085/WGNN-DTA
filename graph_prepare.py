<<<<<<< HEAD
import torch
import esm
import math
import numpy as np
import matplotlib.pyplot as plt
import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm


# data prepare
def protein_graph_construct(proteins, save_dir):
    # Load ESM-1b model
    # torch.set_grad_enabled(False)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    target_graph = {}

    count = 0
    key_list=[]
    for key in proteins:
        key_list.append(key)


    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        # if len(proteins[key]) < 1500:
        #     continue
        # print('=============================================')
        data = []
        pro_id = key
        if os.path.exists(save_dir + pro_id + '.npy'):
            continue
        seq = proteins[key]
        if len(seq) <= 1000:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()
            target_graph[pro_id] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 500
            i = math.ceil(len(seq) / interval)
            # ======================
            # =                    =
            # =                    =
            # =          ======================
            # =          =*********=          =
            # =          =*********=          =
            # ======================          =
            #            =                    =
            #            =                    =
            #            ======================
            # where * is the overlapping area
            # subsection seq contact map prediction
            for s in range(i):
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start

                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                    0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            target_graph[pro_id] = contact_prob_map

        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])
        count += 1
        # # for test
        # print(count, 'of', len(proteins))
        # print('protein id', pro_id)
        # print('seq length:', len(seq))
        # print(target_graph[pro_id].shape)
        # print(len(np.where(target_graph[pro_id] >= 0.5)[0]))
        # plt.matshow(target_graph[pro_id][: len(seq), : len(seq)])
        # plt.title(pro_id)
        # plt.savefig('test/' + pro_id + '.png')
        # print('=============================================')


if __name__ == '__main__':
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


    # # for test
    dataset = 'davis'
    if dataset in ['kiba', 'davis']:
        proteins = json.load(open('data/' + dataset + '/proteins.txt'), object_pairs_hook=OrderedDict)
    elif dataset in ['human', 'celegans']:
        proteins_file = os.path.join('data', dataset, 'proteins.txt')
        if not os.path.exists(proteins_file):
            proteins = {}
            data_file = os.path.join('data', dataset, 'original', 'data.txt')
            seq_list = []
            count = 1
            for line in open(data_file, 'r').readlines():
                if line.strip() == '':
                    continue
                # print(line)
                arrs = line.split(' ')
                entry = [arrs[0], arrs[1], arrs[2]]
                if entry[1] not in seq_list:
                    # print('involve.')
                    key_str = 'prot' + str(count)
                    assert key_str not in proteins.keys()
                    proteins[key_str] = entry[1]
                    seq_list.append(entry[1])
                    count += 1
                # print(entry)
            save_obj(proteins, os.path.join(proteins_file))
        proteins = load_obj(os.path.join(proteins_file))
    print('dataset:', dataset)
    print(len(proteins))

    save_dir = 'pre_process/' + dataset + '/distance_map/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    protein_graph_construct(proteins, save_dir)
=======
import torch
import esm
import math
import numpy as np
import matplotlib.pyplot as plt
import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm


# data prepare
def protein_graph_construct(proteins, save_dir):
    # Load ESM-1b model
    # torch.set_grad_enabled(False)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    target_graph = {}

    count = 0
    key_list=[]
    for key in proteins:
        key_list.append(key)


    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        # if len(proteins[key]) < 1500:
        #     continue
        # print('=============================================')
        data = []
        pro_id = key
        if os.path.exists(save_dir + pro_id + '.npy'):
            continue
        seq = proteins[key]
        if len(seq) <= 1000:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()
            target_graph[pro_id] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 500
            i = math.ceil(len(seq) / interval)
            # ======================
            # =                    =
            # =                    =
            # =          ======================
            # =          =*********=          =
            # =          =*********=          =
            # ======================          =
            #            =                    =
            #            =                    =
            #            ======================
            # where * is the overlapping area
            # subsection seq contact map prediction
            for s in range(i):
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start

                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                    0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            target_graph[pro_id] = contact_prob_map

        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])
        count += 1
        # # for test
        # print(count, 'of', len(proteins))
        # print('protein id', pro_id)
        # print('seq length:', len(seq))
        # print(target_graph[pro_id].shape)
        # print(len(np.where(target_graph[pro_id] >= 0.5)[0]))
        # plt.matshow(target_graph[pro_id][: len(seq), : len(seq)])
        # plt.title(pro_id)
        # plt.savefig('test/' + pro_id + '.png')
        # print('=============================================')


if __name__ == '__main__':
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


    # # for test
    dataset = 'davis'
    if dataset in ['kiba', 'davis']:
        proteins = json.load(open('data/' + dataset + '/proteins.txt'), object_pairs_hook=OrderedDict)
    elif dataset in ['human', 'celegans']:
        proteins_file = os.path.join('data', dataset, 'proteins.txt')
        if not os.path.exists(proteins_file):
            proteins = {}
            data_file = os.path.join('data', dataset, 'original', 'data.txt')
            seq_list = []
            count = 1
            for line in open(data_file, 'r').readlines():
                if line.strip() == '':
                    continue
                # print(line)
                arrs = line.split(' ')
                entry = [arrs[0], arrs[1], arrs[2]]
                if entry[1] not in seq_list:
                    # print('involve.')
                    key_str = 'prot' + str(count)
                    assert key_str not in proteins.keys()
                    proteins[key_str] = entry[1]
                    seq_list.append(entry[1])
                    count += 1
                # print(entry)
            save_obj(proteins, os.path.join(proteins_file))
        proteins = load_obj(os.path.join(proteins_file))
    print('dataset:', dataset)
    print(len(proteins))

    save_dir = 'pre_process/' + dataset + '/distance_map/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    protein_graph_construct(proteins, save_dir)
>>>>>>> 496001edb823488862f0910433051342919e21f0
