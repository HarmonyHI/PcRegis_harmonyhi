import torch
from utils import to_md5

disable_keypoint = True
disable_LRF = True
skip_rt = True
gaussian_noise = False
unseen = False
shuffle = True
drop_last = True
need_save_net = False
last_loss = None
now_epoch = None
epochs = 100

train_split_block = 1
train_split_number = 40960
train_sample_num = train_split_block * train_split_number
train_batch_size = 2048

test_split_block = 512
test_split_number = 16
test_sample_num = test_split_block * test_split_number
test_batch_size = 2

colorful_output_version = 0

xyz_dis_limit = 0.8
feat_dis_limit = 9999999999
knn_num = 64
knn_dis = 0.8
knn_max_empty_num = 5
top_k_num = 2048
num_workers = 2
learn_rate = 0.1
weight_decay = 0.001
milestones = [40, 85]
gamma = 0.1
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
numpy_format = 'float32'
tensor_format = torch.float32
reduction = 'mean'

junk_dir = "F:\\code_junk\\"
src_path = "F:\\WHU\\WHU_dir\\6-Campus\\9.txt"
tgt_path = "F:\\WHU\\WHU_dir\\6-Campus\\10.txt"
rt_path = "F:\\WHU\\WHU_dir\\6-Campus\\9-10\\transformation.txt"
src_cache_name = to_md5(src_path[-9::] + str(knn_num) + str(knn_dis) + str(knn_max_empty_num) + str(xyz_dis_limit))
tgt_cache_name = to_md5(tgt_path[-9::] + str(knn_num) + str(knn_dis) + str(knn_max_empty_num) + str(xyz_dis_limit))
net_path = junk_dir + "saved_net.torchnet"
src_knn_path = junk_dir + f"knn_{src_cache_name}.npy"
tgt_knn_path = junk_dir + f"knn_{tgt_cache_name}.npy"
src_corr_idx_path = junk_dir + f"corr_{src_cache_name}.npy"
tgt_corr_idx_path = junk_dir + f"corr_{tgt_cache_name}.npy"
