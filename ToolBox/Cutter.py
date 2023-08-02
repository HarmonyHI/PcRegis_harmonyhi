import open3d as o3d
import os
from tqdm import tqdm
from support.IOStream import IOStream


def cutter(ori_path, goal_path, start, end):
    pcd = o3d.io.read_point_cloud(ori_path, format="xyz")
    if os.path.isfile(goal_path):
        os.remove(goal_path)
    io = IOStream(goal_path)
    for point in tqdm(pcd.points[start:end]):
        io.cprint(str(point).replace("[", "").replace("]", ""))


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    cutter(PARENT_DIR + '/PointCloudDir/' + 'down_sample_pointcloud_bg.txt',
           PARENT_DIR + '/PointCloudDir/' + 'down_sample_pointcloud.txt', 30000, 34399)
