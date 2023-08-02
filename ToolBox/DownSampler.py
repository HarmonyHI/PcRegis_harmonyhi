import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame

from open_las import pointcloud
import os


def voxel_filter(point_cloud, leaf_size, mode='random'):
    # 首先建立voxel grid
    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)
    D_x = (x_max - x_min) / leaf_size
    D_y = (y_max - y_min) / leaf_size
    D_z = (z_max - z_min) / leaf_size

    # 获取每个点在格子中的位置
    point_cloud = np.asarray(point_cloud)
    h = []
    for i in range(point_cloud.shape[0]):
        h_x = np.floor((point_cloud[i][0] - x_min) / leaf_size)
        h_y = np.floor((point_cloud[i][1] - y_min) / leaf_size)
        h_z = np.floor((point_cloud[i][2] - z_min) / leaf_size)
        H = h_x + h_y * D_x + h_z * D_x * D_y
        h.append(H)

    # 对所有点根据其所在格子位置进行排序
    h = np.asarray(h)
    voxel_index = np.argsort(h)
    h_sort = h[voxel_index]

    # random
    if mode == 'random':
        filtered_points = []
        index_begin = 0
        for i in range(len(voxel_index) - 1):
            if h_sort[i] == h_sort[i + 1]:
                continue
            else:
                point_index = voxel_index[index_begin:(i + 1)]
                random_index = np.random.choice(point_index)
                random_choice = point_cloud[random_index]
                filtered_points.append(random_choice)
                index_begin = i
    # centroid
    if mode == 'centroid':
        filtered_points = []
        index_begin = 0
        for i in range(len(voxel_index) - 1):
            if h_sort[i] == h_sort[i + 1]:
                continue
            else:
                point_index = voxel_index[index_begin:(i + 1)]
                filtered_points.append(np.mean(point_cloud[point_index], axis=0))
                index_begin = i
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def deal(num):
    if 0 <= num <= 9:
        return "140706_"+"0"+str(num)
    else:
        return "140706_"+str(num)


# 读取点云数据
def main_one(i,path,ori_path):
    print("File" + str(i))
    ori_path = ori_path + deal(i) + '.las'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + str(i) + ".txt"
    file = open(path, 'w')
    raw_point_cloud_matrix = pointcloud.load_pointcloud(i,ori_path)
    raw_point_cloud = DataFrame(raw_point_cloud_matrix)
    raw_point_cloud.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(raw_point_cloud)

    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.1, mode='centroid')

    for j in range(filtered_cloud.shape[0]):
        x = filtered_cloud[j, 0]
        y = filtered_cloud[j, 1]
        z = filtered_cloud[j, 2]
        message = str(float(x)) + "  " + str(float(y)) + "  " + str(float(z))
        file.write(message)
        file.write("\r\n")
    file.close()


def main():
    sig_pth = "\\5-Park\\"
    length = 32
    alist = range(1,length+1)
    abs_path = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(abs_path)
    path = PARENT_DIR + "\\PointCloudDir\\WHU_dir"
    path = path + sig_pth
    ori_path = "C:/Users/Chao/Desktop/WHU/WHU-TLS/" + sig_pth
    ori_path = ori_path + '/1-RawPointCloud/'
    for i in alist:
        main_one(i,path,ori_path)


if __name__ == "__main__":
    main()
