import traceback

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import random

import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

sys.path.append('/')
from dta_components.utils import *

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()

    mol_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))

    edges = []
    bond_type_np = np.zeros((mol_size, mol_size))
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_type_np[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondTypeAsDouble()
        bond_type_np[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondTypeAsDouble()
        # bond_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    # print('@@@@@@@@@@@@@@@@@')
    # print(np.array(edges).shape,'edges')
    # print(np.array(g).shape,'g')

    mol_adj = np.zeros((mol_size, mol_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    # print(np.array(mol_adj).shape,'mol_adj')
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    bond_edge_index = []
    bond_type = []
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        bond_edge_index.append([i, j])
        bond_type.append(bond_type_np[i, j])
    # print(bond_edge_index)
    # print('smile_to_graph')
    # print('mol_features',np.array(mol_features).shape)
    # print('bond_edge_index',np.array(bond_edge_index).shape)
    # print('bond_type',np.array(bond_type).shape)
    return mol_size, mol_features, bond_edge_index, bond_type
    # return mol_size, mol_features, bond_edge_index


# target sequence to target graph
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    # print('***',(os.path.abspath(os.path.join(distance_dir, target_key + '.npy'))))
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    # print(len(index_row))
    # print(len(index_col))
    # print(len(index_row_))
    # print(len(index_col_))
    # print(distance_map.shape)
    # print((len(index_row) * 1.0) / (distance_map.shape[0] * distance_map.shape[1]))
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature(target_sequence)
    # residue_distance = np.array(target_edge_distance)  # consistent with edge
    # print('target_feature', target_feature.shape)
    # print(target_edge_index)
    # print('target_edge_index', np.array(target_edge_index).shape)
    # print('residue_distance', residue_distance.shape)
    # return target_size, target_feature, residue_edge_index, residue_distance
    return target_size, target_feature, target_edge_index, target_edge_distance


# data write to csv file
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_DTA_dataset(dataset='davis'):
    # load dataset
    dataset_dir = os.path.join('data', dataset)
    # drug smiles
    ligands = json.load(open(os.path.join(dataset_dir, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    # protein sequences
    proteins = json.load(open(os.path.join(dataset_dir, 'proteins.txt')), object_pairs_hook=OrderedDict)
    # affinity
    affinity = pickle.load(open(os.path.join(dataset_dir, 'Y'), 'rb'), encoding='latin1')
    # dataset divide
    train_fold_origin = json.load(open(os.path.join(dataset_dir, 'folds', 'train_fold_setting1.txt')))
    test_set = json.load(open(os.path.join(dataset_dir, 'folds', 'test_fold_setting1.txt')))
    train_set = [tt for t in train_fold_origin for tt in t]

    # load protein feature and predicted distance map
    process_dir = os.path.join('./', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')  # numpy .npy file

    # dataset process
    drugs = []  # rdkit entity
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    drug_smiles = []  # smiles
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    # consist with deepDTA
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    # dataset content load
    opts = ['train', 'test']
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_set], cols[train_set]
            train_set_entries = []
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_set_entries.append(ls)

            csv_file = dataset + '_' + opt + '.csv'
            data_to_csv(csv_file, train_set_entries)

        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[test_set], cols[test_set]
            test_set_entries = []
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                test_set_entries.append(ls)

            csv_file = dataset + '_' + opt + '.csv'
            data_to_csv(csv_file, test_set_entries)

    print('dataset:', dataset)
    # print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    print('train entries:', len(train_set))
    print('test entries:', len(test_set))

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        g_t = sequence_to_graph(key, proteins[key], pro_distance_dir)
        target_graph[key] = g_t

    # create smile graph
    smile_graph = {}
    for i in tqdm(range(len(drugs))):
        smile = drugs[i]
        g_d = smile_to_graph(smile)
        smile_graph[smile] = g_d

    # for test
    # for smile in drugs:
    #     g_d = smile_to_graph(smile)
    #     smile_graph[smile] = g_d
    # count += 1
    # print(count, len(drugs), 'drugs')
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_csv = dataset + '_train' + '.csv'
    df_train_set = pd.read_csv(train_csv)
    train_drugs, train_prot_keys, train_Y = list(df_train_set['drug_smiles']), list(
        df_train_set['target_key']), list(df_train_set['affinity'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs,
                               target_key=train_prot_keys, y=train_Y, smile_graph=smile_graph,
                               target_graph=target_graph)

    # test dataset construct
    test_csv = dataset + '_test.csv'
    df_test_fold = pd.read_csv(test_csv)
    test_drugs, test_prots_keys, test_Y = list(df_test_fold['drug_smiles']), list(
        df_test_fold['target_key']), list(df_test_fold['affinity'])
    test_drugs, test_prots_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prots_keys), np.asarray(
        test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs,
                              target_key=test_prots_keys, y=test_Y, smile_graph=smile_graph,
                              target_graph=target_graph)
    return train_dataset, test_dataset

def create_CPI_dataset(dataset='human', ratio_n=1):
    # load dataset
    data_file = os.path.join('data', dataset, 'original', 'data.txt')
    proteins_file = os.path.join('data', dataset, 'proteins.txt')
    if not os.path.exists(proteins_file):
        proteins = {}
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
    proteins_rev = {v: k for k, v in proteins.items()}

    all_entries = []
    all_p_entries = []
    all_n_entries = []
    drug_smiles = []
    pro_seqs = []
    pro_keys = []

    for line in open(data_file, 'r').readlines():
        if line.strip() == '':
            continue
        arrs = line.split(' ')
        entry = [arrs[0], proteins_rev[arrs[1]], float(arrs[2])]
        drug_smiles.append(arrs[0])
        pro_seqs.append(arrs[1])
        pro_keys.append(proteins_rev[arrs[1]])
        all_entries.append(entry)
        if float(arrs[2]) == 1:
            all_p_entries.append(entry)
        if float(arrs[2]) == 0:
            all_n_entries.append(entry)

    drug_smiles = list(set(drug_smiles))
    pro_seqs = list(set(pro_seqs))
    pro_keys = list(set(pro_keys))

    print('drug number:', len(drug_smiles))
    print('protein number:', len(pro_seqs), len(pro_keys))
    print('number of entries:', len(all_entries))
    print('number of positive entries:', len(all_p_entries))
    print('number of negative entries:', len(all_n_entries))

    # shuffle
    random.shuffle(all_entries)
    random.shuffle(all_p_entries)
    random.shuffle(all_n_entries)

    used_entries = []
    if ratio_n == 1:
        L = max(len(all_p_entries), len(all_n_entries))
        used_entries = all_p_entries[:L] + all_n_entries[:L]
        random.shuffle(used_entries)
        print("number of used entries:", len(used_entries), "number of positive:", L, "number of negative:", L)
    if ratio_n == 3:
        L = int(len(all_n_entries) / 3.0)
        used_entries = all_p_entries[:L] + all_n_entries[:3 * L]
        random.shuffle(used_entries)
        print("number of used entries:", len(used_entries), "number of positive:", L, "number of negative:", 3 * L)
    if ratio_n == 5:
        L = int(len(all_n_entries) / 5.0)
        used_entries = all_p_entries[:L] + all_n_entries[:5 * L]
        random.shuffle(used_entries)
        print("number of used entries:", len(used_entries), "number of positive:", L, "number of negative:", 5 * L)

    # to used all data
    # used_entries = all_entries

    # split training, validation and test sets
    used_entries = np.array(used_entries)
    np.random.seed(1234)
    ratio = 0.8
    n = int(ratio * len(used_entries))
    train_set, dataset_ = used_entries[:n], used_entries[n:]
    ratio = 0.5
    n = int(ratio * len(used_entries))
    dev_set, test_set = dataset_[:n], dataset_[n:]

    process_dir = os.path.join('./', 'pre_process')
    pro_distance_dir = os.path.join(process_dir, dataset, 'distance_map')  # numpy .npy file
    # create target graph
    target_graph = {}
    for i in tqdm(range(len(pro_keys))):
        key = pro_keys[i]
        seq = proteins[key]
        g_t = sequence_to_graph(key, seq, pro_distance_dir)
        target_graph[key] = g_t

    # create smile graph
    smile_graph = {}
    for i in tqdm(range(len(drug_smiles))):
        smile = drug_smiles[i]
        g_d = smile_to_graph(smile)
        smile_graph[smile] = g_d

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_drugs, train_prot_keys, train_Y = np.asarray(train_set)[:, 0], np.asarray(train_set)[:, 1], np.asarray(
        train_set)[:, 2]
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y.astype(float), smile_graph=smile_graph, target_graph=target_graph)
    # valid dataset construct
    dev_drugs, dev_prot_keys, dev_Y = np.asarray(dev_set)[:, 0], np.asarray(dev_set)[:, 1], np.asarray(dev_set)[:, 2]
    dev_dataset = DTADataset(root='data', dataset=dataset + '_' + 'dev', xd=dev_drugs,
                             target_key=dev_prot_keys, y=dev_Y.astype(float), smile_graph=smile_graph,
                             target_graph=target_graph)
    # test dataset construct
    test_drugs, test_prots_keys, test_Y = np.asarray(test_set)[:, 0], np.asarray(test_set)[:, 1], np.asarray(
        test_set)[:, 2]
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs,
                              target_key=test_prots_keys, y=test_Y.astype(float), smile_graph=smile_graph,
                              target_graph=target_graph)
    temp_y = test_Y.astype(float)
    # print(type(temp_y))
    return train_dataset, dev_dataset, test_dataset

def create_CPI_dataset_with_structure(dataset='dude'):
    # load dataset
    dataset_dir = os.path.join('data', 'dud_e','all')
    save_map_dir = os.path.join('pre_process', 'dud_e', 'distance_map')
    dataset_entries_csv = os.path.join('data', 'dud_e', 'data_entries.csv')
    protein_pkl = os.path.join('data', 'dud_e', 'proteins')
    # affinity = load_obj(save_affinity_pkl)
    # drug = load_obj(save_drug_pkl)
    # target = load_obj(save_protein_pkl)
    # df_pdb=pd.read_csv(save_valid_pdb_csv)
    # pdb_list=list(df_train_set['drug_smiles'])
    df_dataset = pd.read_csv(dataset_entries_csv)
    df_dataset.fillna(0, inplace=True)
    dataset_drugs, dataset_prot_seqs, dataset_prot_keys, dataset_Y = list(df_dataset['drug_smiles']), list(
        df_dataset['target_sequence']), list(df_dataset['target_key']), list(df_dataset['affinity'])
    drugs = list(set(dataset_drugs))
    pros = list(set(dataset_prot_keys))
    proteins = load_obj(protein_pkl)
    ratio = 0.6
    random.shuffle(pros)
    train_prots=pros[:int(ratio*len((pros)))]
    test_prots=pros[int(ratio*len((pros))):]
    entries = []
    for smiles, seq, key, y in zip(dataset_drugs, dataset_prot_seqs, dataset_prot_keys, dataset_Y):
        entries.append([smiles, seq, key, y])
    random.shuffle(entries)
    print('all entries:', len(entries))
    # exit(0)

    target_graph = {}
    drug_graph = {}
    mol_error_list=[]
    for p_i in tqdm(range(len(pros))):
        pdb = pros[p_i]
        g_t = sequence_to_graph(pdb, proteins[pdb], save_map_dir)
        target_graph[pdb] = g_t
    for d_i in tqdm(range(len(drugs))):
        smile = drugs[d_i]
        try:
            g_d = smile_to_graph(smile)
            drug_graph[smile] = g_d
        except:
            if smile not in mol_error_list:
                mol_error_list.append(smile)
            print(smile,'error')
            # traceback.print_exc()

    train_entries=[]
    test_entries=[]
    for entry in entries:
        if entry[0] in mol_error_list:
            continue
        if entry[2] in train_prots:
            train_entries.append(entry)
        elif entry[2] in test_prots:
            test_entries.append(entry)
    # valid_entries=[]
    # for entry in entries:
    #     if entry[0] in mol_error_list:
    #         continue
    #     valid_entries.append(entry)
    # ratio = 0.8
    # random.shuffle(valid_entries)
    # train_entries = valid_entries[:int(len(valid_entries)*ratio)]
    # test_entries = valid_entries[int(len(valid_entries)*ratio):]
    # print('used_entries:', len(valid_entries))
    print('train_entries:', len(train_entries),'train proteins:',len(train_prots))
    print('test_entries:', len(test_entries),'test proteins:',len(test_prots))
    # print(np.array(train_entries).shape)
    # print(np.array(test_entries).shape)
    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    # train dataset construct
    train_drugs, train_prot_keys, train_Y = np.asarray(train_entries)[:, 0], np.asarray(train_entries)[:,
                                                                             2], np.asarray(train_entries)[:, 3]
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y.astype(float), smile_graph=drug_graph, target_graph=target_graph)

    # test dataset construct
    test_drugs, test_prots_keys, test_Y = np.asarray(test_entries)[:, 0], np.asarray(test_entries)[:, 2], np.asarray(
        test_entries)[:, 3]
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs, target_key=test_prots_keys,
                              y=test_Y.astype(float), smile_graph=drug_graph, target_graph=target_graph)
    temp_y = test_Y.astype(float)
    # print(type(temp_y))
    return train_dataset, test_dataset


if __name__ == '__main__':
    create_CPI_dataset('human', 1)
