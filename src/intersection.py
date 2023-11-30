import numpy as np


def get_intersect(line1_start, line1_end, line2_start, line2_end):
    h = np.hstack(([line1_start, line1_end, line2_start, line2_end], np.ones((4, 1))))
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)
    if z == 0:
        return float('inf'), float('inf')
    return x / z, y / z


def get_len(line_start, line_end):
    return np.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1])


def get_vector(line):
    x1, y1, x2, y2 = line
    return np.array([x2 - x1, y2 - y1])


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(line1, line2):
    v1 = get_vector(line1)
    v2 = get_vector(line2)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


if __name__ == "__main__":
    print(get_intersect((0, 1), (0, 2), (1, 10), (1, 9)))  # parallel  lines
    print(get_intersect((0, 1), (0, 2), (1, 10), (2, 10)))  # vertical and horizontal lines
    print(get_intersect((0, 1), (1, 2), (0, 10), (1, 9)))  # another line for fun
