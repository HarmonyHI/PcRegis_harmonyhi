import os.path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation


def npmat2euler(mats, seq='zyx'):
    eulers = []
    r = Rotation.from_matrix(mats)
    eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def get(src_path, tgt_path, trans_path, goal_path, overlap):
    try:
        transformation = np.loadtxt(trans_path, 'float32')
    except ValueError:
        print(trans_path + " is broken file,change to transformation.txt")
        trans_path = trans_path.replace("-Dong2018", "")
        try:
            transformation = np.loadtxt(trans_path, 'float32')
        except OSError:
            print(trans_path + " is not found,change to transformation-GT.txt")
            trans_path = trans_path.replace("transformation.txt", "transformation-GT.txt")
            transformation = np.loadtxt(trans_path, 'float32')
        print("Changed successfully")
    rotation = transformation[0:3, 0:3]
    translation = transformation[0:3, 3]
    angle = npmat2euler(rotation).squeeze()
    angle = np.deg2rad(angle)
    src = np.loadtxt(src_path, 'float32')
    tgt = np.loadtxt(tgt_path, 'float32')
    output = {'src': src, 'tgt': tgt, 'angle': angle, 'translation': translation, 'overlap': overlap}
    if not os.path.isdir(os.path.dirname(goal_path)):
        os.makedirs(os.path.dirname(goal_path))
    with open(goal_path, "wb") as tf:
        pickle.dump(output, tf)


def main(plc, frm, to, over_lap):
    a_path = "E:\\File\\Pythoncode\\PointCloud\\MFGNet\\PointCloudDir\\WHU_Overlap"
    b_path = "\\3-GroundTruth"
    c_path = "\\transformation-Dong2018.txt"
    d_path = "E:\\File\\Pythoncode\\PointCloud\\MFGNet\\PointCloudDir\\WHU_dir"
    r_path = "E:\\File\\Pythoncode\\PointCloud\\MFGNet\\PointCloudDir\\WHU_PKL"
    e_path = plc
    f1_path = "\\" + str(frm) + ".txt"
    f2_path = "\\" + str(to) + ".txt"
    g_path = "\\" + str(frm) + "-" + str(to)
    h_path = "\\" + str(frm) + "-" + str(to) + ".pkl"
    get(d_path + e_path + f1_path, d_path + e_path + f2_path, a_path + e_path + b_path + g_path + c_path,
        r_path + e_path + h_path, over_lap)


if __name__ == '__main__':
    place_list = {"\\1-SubwayStation": [[1, 3], [2, 4], [3, 5], [4, 5], [6, 5]],
                  "\\2-HighSpeedRailway": [[2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                  "\\3-Mountain": [[1, 2], [2, 3], [3, 4], [5, 4], [6, 5]],
                  "\\5-Park": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 9], [7, 5], [8, 6], [9, 10], [10, 11],
                               [11, 12], [12, 13], [13, 14], [15, 14], [16, 15], [17, 16], [18, 7], [19, 7], [20, 19],
                               [21, 20], [22, 17], [23, 25], [24, 25], [25, 26], [26, 28], [27, 30], [28, 27], [29, 28],
                               [30, 31], [31, 32], [32, 14]],
                  "\\6-Campus": [[1, 2], [2, 3], [4, 3], [5, 4], [6, 5], [7, 6], [8, 6], [9, 10], [10, 8]],
                  "\\7-Residence": [[1, 6], [2, 3], [3, 4], [4, 5], [6, 5], [7, 5]],
                  "\\8-RiverBank": [[1, 2], [2, 3], [3, 4], [5, 4], [6, 5], [7, 6]],
                  "\\9-HeritageBuilding": [[1, 3], [2, 1], [3, 4], [4, 5], [6, 5], [7, 6], [8, 7], [9, 8]],
                  "\\10-UndergroundExcavation": [[1, 3], [2, 1], [4, 3], [5, 3], [6, 3], [7, 6], [8, 7], [9, 7],
                                                 [10, 9], [11, 10], [12, 10]],
                  "\\11-Tunnel": [[1, 2], [2, 3], [3, 4], [5, 4], [6, 5], [7, 6]],

                  }
    max_length_place = {"\\1-SubwayStation": 6,
                        "\\2-HighSpeedRailway": 8,
                        "\\3-Mountain": 6,
                        "\\5-Park": 32,
                        "\\6-Campus": 10,
                        "\\7-Residence": 7,
                        "\\8-RiverBank": 7,
                        "\\9-HeritageBuilding": 9,
                        "\\10-UndergroundExcavation": 12,
                        "\\11-Tunnel": 7
                        }
    for place in place_list.keys():
        print(f"===============file{place}==============")
        f = open("E:\\File\\Pythoncode\\PointCloud\\MFGNet\\PointCloudDir\\WHU_Overlap" + place + "\\overlap.txt")
        overlap_dict = {}
        for j in range(1, max_length_place[place] + 1, 1):
            for jj in range(j + 1, max_length_place[place] + 1, 1):
                tmp = f.readline().split(" ")
                ori_num = int(tmp[0])
                goal_num = int(tmp[1])
                overlap_value = float(tmp[2])
                overlap_dict[(ori_num, goal_num)] = overlap_value
        for d in place_list[place]:
            try:
                overlap_tmp = overlap_dict[(d[0], d[1])]
            except KeyError:
                print(f"Request overlap {d[0]} to {d[1]} is existed as {d[1]} to {d[0]}, Replaced")
                overlap_tmp = overlap_dict[(d[1], d[0])]
            print(f"Dealing {place} {str(d[0])} to {str(d[1])} whose overlap is {overlap_tmp}")
            main(place, d[0], d[1], overlap_tmp)
