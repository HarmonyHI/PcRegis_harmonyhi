from .head import get_header
from .point import Point
import struct


def get_version(f):
    f.read(24)
    version_major, version_minor = struct.unpack('2B', f.read(2))
    # print(f"las版本:{version_major}.{version_minor}")
    return version_major, version_minor


def load_pointcloud(i,data_dir):
    # print(data_dir)
    f = open(data_dir, 'rb')
    version = get_version(f)
    header = get_header(f, version)
    # print(header.__dict__)
    points = Point(header.x_scale_factor,
                   header.y_scale_factor,
                   header.z_scale_factor,
                   header.x_offset,
                   header.y_offset,
                   header.z_offset,
                   )
    pcd = points.read_point(f, header.offset_to_point_data,
                            header.point_data_record_format,
                            header.number_of_point_records)
    f.close()
    # print(data_dir, "load succeed")
    return pcd

