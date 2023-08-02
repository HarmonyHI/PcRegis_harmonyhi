import hashlib
import os
import random
import sys
from os.path import isdir, isfile
import numpy as np
import torch
from scipy.spatial import kdtree
from tqdm import tqdm
import args


def sp_point_match(src_knn, tgt_knn, src_feat, tgt_feat):
    src_knn_list = []
    tgt_knn_list = []
    src_feat_list = []
    tgt_feat_list = []
    for batch_iter in range(src_feat.shape[0]):
        for point_idx in range(src_feat.shape[1]):
            src_knn_list.append(src_knn[batch_iter][point_idx])
            src_feat_list.append(src_feat[batch_iter][point_idx])
    for batch_iter in range(tgt_feat.shape[0]):
        for point_idx in range(tgt_feat.shape[1]):
            tgt_knn_list.append(tgt_knn[batch_iter][point_idx])
            tgt_feat_list.append(tgt_feat[batch_iter][point_idx])
    tgt_feat_list = torch.stack(tgt_feat_list).detach().numpy()
    src_feat_list = torch.stack(src_feat_list).detach().numpy()
    tree = kdtree.KDTree(tgt_feat_list)
    src_match_list = []
    tgt_match_list = []
    for src_i in range(src_feat_list.__len__()):
        dis, tgt_i = tree.query(src_feat_list[src_i], distance_upper_bound=args.feat_dis_limit)
        if not tgt_i == tgt_feat_list.__len__():
            src_match_list.append(src_knn_list[src_i])
            tgt_match_list.append(tgt_knn_list[tgt_i])
    return torch.stack(src_match_list).unsqueeze(0), torch.stack(tgt_match_list).unsqueeze(0)  # b,n,k,3


def find_match_pair(src_points, tgt_points, R, t):
    src_points = (torch.matmul(R, src_points.permute((0, 2, 1))) + t.unsqueeze(-1)).permute((0, 2, 1))
    src_match_list = []
    tgt_match_list = []
    for batch_iter in range(src_points.shape[0]):
        src_match_batch_list = []
        tgt_match_batch_list = []
        tree = kdtree.KDTree(src_points[batch_iter])
        for i in tqdm(range(tgt_points[batch_iter].__len__())):
            _, match_idx_of_src = tree.query(tgt_points[batch_iter][i], distance_upper_bound=args.xyz_dis_limit)
            if not match_idx_of_src == src_points[batch_iter].__len__():
                src_match_batch_list.append(match_idx_of_src)
                tgt_match_batch_list.append(i)
        src_match_list.append(src_match_batch_list)
        tgt_match_list.append(tgt_match_batch_list)
    return src_match_list, tgt_match_list


def output_color_corr_xyz(data_a, data_b, name, stop=False):
    # data: b,n,k,3
    print("colorful print")
    path_a = args.junk_dir + f"{name}_a.txt"
    path_b = args.junk_dir + f"{name}_b.txt"
    file_a = open(path_a, "w+")
    file_b = open(path_b, "w+")
    color_table = []
    for batch_iter in range(data_a.__len__()):
        batch_color_table = []
        for item_iter in range(data_a[batch_iter].__len__()):
            batch_color_table.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        color_table.append(batch_color_table)
    for batch_iter in range(data_a.__len__()):
        for item_iter in range(data_a[batch_iter].__len__()):
            r, g, b = color_table[batch_iter][item_iter]
            color_str = f"{str(int(r))} {str(int(g))} {str(int(b))}"
            for knn_i in range(args.knn_num + 1):
                a_mess = f"{str(data_a[batch_iter][item_iter][knn_i][0]).replace('tensor(', '').replace(')', '')}" + \
                         f" {str(data_a[batch_iter][item_iter][knn_i][1]).replace('tensor(', '').replace(')', '')}" + \
                         f" {str(data_a[batch_iter][item_iter][knn_i][2]).replace('tensor(', '').replace(')', '')} " \
                         + color_str
                b_mess = f"{str(data_b[batch_iter][item_iter][knn_i][0]).replace(')', '').replace('tensor(', '')}" + \
                         f" {str(data_b[batch_iter][item_iter][knn_i][1]).replace(')', '').replace('tensor(', '')}" + \
                         f" {str(data_b[batch_iter][item_iter][knn_i][2]).replace(')', '').replace('tensor(', '')} " \
                         + color_str
                file_a.write(a_mess)
                file_b.write(b_mess)
                file_a.write("\n")
                file_b.write("\n")
    print("colorful finished")
    if stop:
        sys.exit(0)


def to_md5(origin):
    md5 = hashlib.md5()
    md5.update(origin.encode())
    return md5.hexdigest()[0:15]


def ensure_path(path):
    if not isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))


def ensure_delete(path):
    if isfile(path):
        os.remove(path)


def random_down_sample(point_cloud, goal_num):
    list0 = random.sample(range(0, point_cloud.__len__()), goal_num)
    return point_cloud[list0]


def write_out_txt(data, goal_path, stop=False):
    shape = 1
    shape_list = data.shape[0:-1]
    for s in shape_list:
        shape = shape * s
    data = data.reshape(shape, data.shape[-1])
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()
    if os.path.isfile(goal_path):
        os.remove(goal_path)
    np.savetxt(goal_path, data)
    if stop:
        sys.exit(0)


def neighbour_maker(pcd, knn_num):
    tree = kdtree.KDTree(pcd)
    total_knn = []
    for center_point in tqdm(pcd):
        _, neighbour_arr = tree.query(center_point, knn_num + 1, distance_upper_bound=args.knn_dis)
        empty_goal = 0
        for i in range(neighbour_arr.__len__()):
            if neighbour_arr[i] == pcd.__len__():
                neighbour_arr[i] = neighbour_arr[0]
                empty_goal = empty_goal + 1
        if empty_goal <= args.knn_max_empty_num:
            total_knn.append(np.asarray(neighbour_arr))
    return np.asarray(total_knn)


def get_k_idx(score, k):
    score, score_idx = torch.sort(score, dim=-1, descending=True)
    if args.disable_keypoint:
        score_idx = score_idx[:, torch.randperm(score_idx.size(1))]
    top_k_idx = score_idx[:, 0:k]
    return top_k_idx


def tensor_rebuild(data, idx):
    # data shape: numpy_B,N,K,3 or tensor_B,N,128
    # idx shape: B,TOP_K
    # return shape: B,TOP_K,K,3 or B,TOP_K,128
    if str(type(data)) == "<class 'torch.Tensor'>":
        tmp_data_block = []
        for i in range(data.shape[0]):
            tmp_data_block.append(data[i, idx[i]])
        tmp_data_block = torch.stack(tmp_data_block)
        return tmp_data_block
    else:
        tmp_data_block = []
        for i in range(data.shape[0]):
            tmp_data_block.append(data[i, idx[i]])
        tmp_data_block = np.asarray(tmp_data_block, dtype=args.numpy_format)
        return tmp_data_block


def list_rebuild(data, idx):
    # data shape: numpy_B,N,* or tensor_B,N,*
    # idx shape: list_B,VARIABLE
    # return shape: list_B,tensor(VARIABLE,K,3) or list_B,numpy(VARIABLE,K,3)
    total_list = []
    for batch_iter in range(idx.__len__()):
        batch_tmp_data = []
        for item_iter in idx[batch_iter]:
            batch_tmp_data.append(data[batch_iter][item_iter])
        total_list.append(torch.stack(batch_tmp_data))
    return total_list


def seed_everything():
    seed = random.randint(0, 100)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
