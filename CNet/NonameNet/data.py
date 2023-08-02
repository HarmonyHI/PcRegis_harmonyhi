import os.path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import args
import utils
from utils import neighbour_maker, ensure_path


def load_rt_matrix(path, data_format):
    try:
        matrix = np.loadtxt(path, dtype=data_format)  # [n, 3]
    except OSError:
        ensure_path(path)
        print(f"{path} is not found, please check")
        sys.exit(1)
    rotate_matrix = matrix[0:3, 0:3]
    trans_matrix = matrix[0:3, 3]
    return rotate_matrix, trans_matrix


def load_point_cloud(path, knn_path, knn_num, data_format):
    if os.path.isfile(path):
        pc = np.loadtxt(path, dtype=data_format)  # [n, 3]
    else:
        print(f"{path} is not found, please check")
        sys.exit(1)
    if os.path.isfile(knn_path):
        pc_knn_idx = np.load(knn_path, allow_pickle=True)
        print("knn loaded from file")
    else:
        print("not found knn, creating")
        pc_knn_idx = neighbour_maker(pc, knn_num)
        np.save(knn_path, pc_knn_idx)
        print("knn saved")
    pc_knn = pc[pc_knn_idx]  # [n, knn_num + 1, 3]
    return pc_knn


def get_corr_node(src_knn, tgt_knn, rot, trans):
    """
    Input:
        src_knn: numpy [n-src, k-nearest-neighbour, 3]
        tgt_knn: numpy [n-tgt, k-nearest-neighbour, 3]
        rot: [3, 3]
        trans: [3, 1]
    Return:
        src_knn_corr, tgt_knn_corr: tensor [n-paired, 2, k-nearest-neighbour, 3]
    """
    src_knn = torch.tensor(src_knn).unsqueeze(0)
    tgt_knn = torch.tensor(tgt_knn).unsqueeze(0)
    rot = torch.tensor(rot)
    trans = torch.tensor(trans)
    if os.path.isfile(args.src_corr_idx_path) and os.path.isfile(args.tgt_corr_idx_path):
        src_knn_corr_idx = np.load(args.src_corr_idx_path, allow_pickle=True)
        tgt_knn_corr_idx = np.load(args.tgt_corr_idx_path, allow_pickle=True)
        print("corr loaded from file")
    else:
        print("finding corr node")
        src_xyz = src_knn[:, :, 0, :]
        tgt_xyz = tgt_knn[:, :, 0, :]
        src_knn_corr_idx, tgt_knn_corr_idx = utils.find_match_pair(src_xyz, tgt_xyz, rot, trans)
        np.save(args.src_corr_idx_path, src_knn_corr_idx)
        np.save(args.tgt_corr_idx_path, tgt_knn_corr_idx)
        print("corr node saved")
    src_knn_corr = utils.tensor_rebuild(src_knn, src_knn_corr_idx)[0, :, :, :]
    tgt_knn_corr = utils.tensor_rebuild(tgt_knn, tgt_knn_corr_idx)[0, :, :, :]
    corr_node = torch.stack([src_knn_corr, tgt_knn_corr]).permute((1, 0, 2, 3))
    return corr_node


class WhuDataset(Dataset):
    def __init__(self, gaussian_noise, unseen, is_train):
        super(WhuDataset, self).__init__()
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.raw_src_knn = load_point_cloud(args.src_path, args.src_knn_path, args.knn_num, args.numpy_format)
        self.raw_tgt_knn = load_point_cloud(args.tgt_path, args.tgt_knn_path, args.knn_num, args.numpy_format)
        self.rot_matrix, self.trans_matrix = load_rt_matrix(args.rt_path, args.numpy_format)
        corr_node = get_corr_node(self.raw_src_knn, self.raw_tgt_knn, self.rot_matrix, self.trans_matrix)
        if is_train:
            corr_node = utils.random_down_sample(corr_node, args.train_sample_num)
            corr_node = corr_node.reshape((args.train_split_number, args.train_split_block,
                                           corr_node.shape[1], corr_node.shape[2], corr_node.shape[3]))
            self.src_knn = corr_node[:, :, 0, :, :]
            self.tgt_knn = corr_node[:, :, 1, :, :]
        else:
            corr_node = utils.random_down_sample(corr_node, args.test_sample_num)
            corr_node = corr_node.reshape((args.test_split_number, args.test_split_block,
                                           corr_node.shape[1], corr_node.shape[2], corr_node.shape[3]))
            self.src_knn = corr_node[:, :, 0, :, :]
            self.tgt_knn = corr_node[:, :, 1, :, :]
            self.src_knn = self.src_knn[:, torch.randperm(self.src_knn.size(1)), :, :]
            self.tgt_knn = self.tgt_knn[:, torch.randperm(self.tgt_knn.size(1)), :, :]

    def __getitem__(self, item):
        return self.src_knn[item], self.tgt_knn[item], self.rot_matrix, self.trans_matrix

    def __len__(self):
        if self.src_knn.shape[0] < self.tgt_knn.shape[0]:
            return self.src_knn.shape[0]
        return self.tgt_knn.shape[0]
