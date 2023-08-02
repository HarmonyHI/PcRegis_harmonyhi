import math
import args
from args import kdtree_dim
import torch.nn.functional as nn_func


class KDtree:
    @staticmethod
    def distance_squared(point1, point2):
        return nn_func.pairwise_distance(point1[0:-1], point2[0:-1])

    def closest_point(self, all_points, new_point):
        best_point = None
        best_distance = None
        for current_point in all_points:
            current_distance = self.distance_squared(new_point, current_point)
            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_point = current_point
        return best_point

    def build_kdtree(self, data, depth=0):
        n = len(data)
        if n <= 0:
            return None
        axis = depth % kdtree_dim
        sorted_points = sorted(data, key=lambda point: point[axis])
        return {
            'point': sorted_points[n // 2],
            'left': self.build_kdtree(sorted_points[:n // 2], depth + 1),
            'right': self.build_kdtree(sorted_points[n // 2 + 1:], depth + 1)
        }

    def kdtree_naive_closest_point(self, root, point, depth=0, best=None):
        if root is None:
            return best
        axis = depth % kdtree_dim
        if best is None or self.distance_squared(point, best) > self.distance_squared(point, root['point']):
            next_best = root['point']
        else:
            next_best = best
        if point[axis] < root['point'][axis]:
            next_branch = root['left']
        else:
            next_branch = root['right']
        return self.kdtree_naive_closest_point(next_branch, point, depth + 1, next_best)

    def closer_distance(self, pivot, p1, p2):
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        d1 = self.distance_squared(pivot, p1)
        d2 = self.distance_squared(pivot, p2)
        if d1 < d2:
            return p1
        else:
            return p2

    def kdtree_closest_point(self, root, point, depth=0):
        if root is None:
            return None
        axis = depth % kdtree_dim
        if point[axis] < root['point'][axis]:
            next_branch = root['left']
            opposite_branch = root['right']
        else:
            next_branch = root['right']
            opposite_branch = root['left']
        best = self.closer_distance(point, self.kdtree_closest_point(next_branch, point, depth + 1), root['point'])
        if self.distance_squared(point, best) > (point[axis] - root['point'][axis]) ** 2:
            best = self.closer_distance(point, self.kdtree_closest_point(opposite_branch, point, depth + 1), best)
        return best

    def create_tree(self, points, pivot):
        for i in range(len(points)):
            points[i] = (*(points[i]), i)
        pivot = (*pivot, 0)
        kdtree = self.build_kdtree(points)
        found = self.kdtree_closest_point(kdtree, pivot)
        if math.sqrt(self.distance_squared(pivot, found)) > args.max_match_limit:
            return None
        return found
