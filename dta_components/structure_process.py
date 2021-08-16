import random

import numpy as np
import Bio.PDB
import esm
import torch
import math
import os
from tqdm import tqdm
import warnings
from Bio import BiopythonWarning
from rdkit import Chem
import networkx as nx
import json, pickle
import traceback

warnings.simplefilter('ignore', BiopythonWarning)

residue_dic = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LLE': 'L',
               'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
               'TRP': 'W', 'TYR': 'Y'}


def contact_map_acc_cal(map_1, map_2):
    temp_map = map_1 - map_2
    col, row = np.where(temp_map == 0)
    return len(col) * 1.0 / (map_1.shape[0] * map_1.shape[1])


# def calc_residue_dist(residue_one, residue_two):
#     """Returns the C-alpha distance between two residues"""
#     diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
#     return np.sqrt(np.sum(diff_vector * diff_vector))
#
#
# def calc_dist_matrix(chain_one, chain_two):
#     """Returns a matrix of C-alpha distances between two chains"""
#     answer = np.zeros((len(chain_one), len(chain_two)), np.float)
#     for row, residue_one in enumerate(chain_one):
#         for col, residue_two in enumerate(chain_two):
#             answer[row, col] = calc_residue_dist(residue_one, residue_two)
#     return answer


# def get_contact_map_from_pdb(pdb_code, pdb_file):
#     structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_file)
#     model = structure[0]
#     dist_matrix = calc_dist_matrix(model, model)
#     contact_map = dist_matrix < 8.0
#     print(contact_map.shape)
#     return contact_map


def get_seq_and_map_from_pdb(pdb_code, pdb_file):
    seq_1l = []
    seq_3l = []
    residue_coors = []
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_file)
    model = structure[0]
    # print(model.get_residues())
    for residue in model.get_residues():
        if residue.get_resname() in residue_dic.keys():
            try:
                coors = residue['CA'].coord
                res_name = residue.get_resname()
                residue_coors.append(coors)
                seq_1l.append(residue_dic[res_name])
                seq_3l.append(res_name)
            except:
                print(residue.get_resname(), 'error.')
                # traceback.print_exc()

    # print(seq_1l)
    # print(seq_3l)
    # print(len(seq_1l))
    # print(len(residue_coors))
    seq_1l = ''.join(seq_1l)
    # print(pdb_code)
    # print(seq_1l)
    seq_length = len(seq_1l)

    distance_map = np.zeros((seq_length, seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            distance_map[i, j] = np.linalg.norm(residue_coors[i] - residue_coors[j])

    # distance_map += np.eye(seq_length)
    # distance_map = np.where(distance_map <= 8, 1.0 / distance_map, 0)
    # print(pdb_code, len(seq_1l), distance_map.shape)
    return seq_1l, distance_map

    # polypeptide_builder = Bio.PDB.CaPPBuilder()
    # counter = 1
    # for polypeptide in polypeptide_builder.build_peptides(model):
    #     seq = polypeptide.get_sequence()
    #     print(f"Sequence: {counter}, Length: {len(seq)}")
    #     print(seq)
    #     counter += 1


def contact_map_predict(pdb_code, seq, model, batch_converter):
    data = []
    data.append((pdb_code, seq))
    if len(seq) <= 1000:
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        contact_prob_map = results["contacts"][0].numpy()
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
            temp_data.append((pdb_code, temp_seq))
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
    return contact_prob_map


def compare_single(pdb_code, pdb_file, model, batch_converter):
    seq, contact_map_t = get_seq_and_map_from_pdb(pdb_code, pdb_file)
    contact_map_p = contact_map_predict(pdb_code, seq, model, batch_converter)
    contact_map_p = np.where(contact_map_p >= 0.5, 1, 0)
    acc = contact_map_acc_cal(contact_map_t, contact_map_p)
    return acc, seq


def compare(pdb_dir, pdb_list):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    # pdb_list = ['1A4W']
    acc_list = []
    acc_small_list = []
    acc_medium_list = []
    acc_large_list = []
    for i in tqdm(range(len(pdb_list))):
        pdb = pdb_list[i]
        pro_file = os.path.join(pdb_dir, pdb, pdb + '_protein.pdb')
        if not os.path.exists(pro_file):
            print(pdb, 'not exist.')
            continue
        temp_acc, temp_seq = compare_single(pdb, pro_file, model, batch_converter)
        acc_list.append(temp_acc)
        if len(temp_seq) <= 1000:
            acc_small_list.append(temp_acc)
        elif len(temp_seq) <= 2000:
            acc_medium_list.append(temp_acc)
        else:
            acc_large_list.append(temp_acc)
    print(np.array(acc_list).shape, len(pdb_list))
    # print(acc_list)
    print('all:', np.average(np.array(acc_list)), len(acc_list))
    print('small:', np.average(np.array(acc_small_list)), len(acc_small_list))
    print('medium:', np.average(np.array(acc_medium_list)), len(acc_medium_list))
    print('large:', np.average(np.array(acc_large_list)), len(acc_large_list))


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')

def save_data_from_dude():
    dataset_dir = os.path.join('../data', 'dud_e', 'all')
    save_map_dir = os.path.join('../pre_process', 'dud_e', 'distance_map')
    dataset_entries_csv = os.path.join('../data', 'dud_e', 'data_entries.csv')
    # save_drug_pkl = os.path.join('../pre_process', 'dud_e', 'drug')
    save_protein_pkl = os.path.join('../data', 'dud_e', 'proteins')
    if not os.path.exists(save_map_dir):
        os.makedirs(save_map_dir)

    pdb_list = os.listdir(dataset_dir)
    print('protein structure entries:', len(pdb_list))  # 102
    target = {}
    used_entries = []
    mol_list = []

    for i in tqdm(range(len(pdb_list))):
        pdb = pdb_list[i]
        try:
            # process protein
            protein_file = os.path.join(dataset_dir, pdb, 'receptor.pdb')
            seq, distance_map = get_seq_and_map_from_pdb(pdb, protein_file)
            target[pdb] = seq

            # print(target_size, target_feature, target_edge_index, target_edge_distance)

            # process molecule
            active_mol_file = os.path.join(dataset_dir, pdb, 'actives_final.ism')
            decoys_mol_file = os.path.join(dataset_dir, pdb, 'decoys_final.ism')
            count = 0  # our server memory can not support too many molecules, so set a limitation number with 500 for negative and positive samples
            for line in open(active_mol_file, 'r').readlines():
                if line.strip() != '':
                    arr = line.split()
                    temp_active_mol_smiles = arr[0]
                    if temp_active_mol_smiles not in mol_list:
                        mol_list.append(temp_active_mol_smiles)
                    used_entries.append([temp_active_mol_smiles, seq, pdb, 1])
                    count += 1
                    if count >= 200:
                        break
            count = 0
            for line in open(decoys_mol_file, 'r').readlines():
                if line.strip() != '':
                    arr = line.split()
                    temp_decoys_mol_smiles = arr[0]
                    if temp_decoys_mol_smiles not in mol_list:
                        mol_list.append(temp_decoys_mol_smiles)
                    used_entries.append([temp_decoys_mol_smiles, seq, pdb, 0])
                    count += 1
                    if count >= 200:
                        break

            np.save(os.path.join(save_map_dir, pdb + '.npy'), distance_map)
        except:
            traceback.print_exc()
            print(pdb, 'error.')
            print('entries:', len(used_entries), len(mol_list))

    save_obj(target, save_protein_pkl)
    # save_obj(affinity, save_affinity_pkl)
    data_to_csv(dataset_entries_csv, used_entries)
    print('all entries:', len(used_entries))


if __name__ == '__main__':
    # just used for a test
    # print(len(residue_dic.keys()))
    # get_seq_and_map_from_pdb('aa2ar', 'receptor.pdb')
    # get_contact_map_from_pdb('1A4W', '1a4w_protein.pdb')

    # for the map accuracy test
    # pdb_list = os.listdir('data/pdbbind2019/refined-set')
    # compare('data/pdbbind2019/refined-set', pdb_list)
    # save_data_from_pdbbind()

    # save the map and used for train and test
    save_data_from_dude()
