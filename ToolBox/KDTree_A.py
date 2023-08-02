import open3d as o3d
import os
from tqdm import tqdm
from support.IOStream import IOStream


def near_find(ori_path, goal_path, ran):
    pcd = o3d.io.read_point_cloud(ori_path, format="xyz")
    pc_tree = o3d.geometry.KDTreeFlann(pcd)
    if os.path.isfile(goal_path):
        os.remove(goal_path)
    io = IOStream(goal_path)
    for point in tqdm(pcd.points):
        goal_pos = [*[*pc_tree.search_knn_vector_3d(point, ran)][1]][1::]
        for i in goal_pos:
            io.cprint(str(pcd.points[i]).replace("[", "").replace("]", ""))


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    near_find(PARENT_DIR + '/PointCloudDir/' + 'down_sample_pointcloud.txt',
              PARENT_DIR + '/PointCloudDir/' + 'nearest_point.txt', 100)
