import argparse
import json
import os
import time

from tqdm import tqdm

import cv2
import numpy as np

from intersection import get_intersect


def write_to_json(path, t_dict):
    with open(path + '/' + 'answers.json', 'w') as json_file:
        prep_data = json.dumps(t_dict)
        json_file.write(prep_data)


def convert_to_gray(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def get_edges(img, sigma):
    """
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    """
    median = np.median(img)
    low_threshold = int(max(0, (1.0 - sigma) * median))
    high_threshold = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(img, low_threshold, high_threshold)


def get_hough_lines(img):
    lines = cv2.HoughLinesP(img,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=100,
                            maxLineGap=50)
    return lines


def get_lines(img, sigma):
    img = get_edges(convert_to_gray(img), sigma)
    lines = get_hough_lines(img)
    if lines is not None:
        return [line[0] for line in lines]
    else:
        return []


def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            try:
                a, b = get_intersect((line1[0], line1[1]), (line1[2], line1[3]),
                                     (line2[0], line2[1]), (line2[2], line2[3]))
                if (a, b) != (float('inf'), float('inf')):
                    intersections.append((a, b))
            except Exception as ex:
                print('intersections\n')
                print(ex)
    return intersections


def find_vp(intersections, img_shape):
    if intersections:
        x_array = [intersection[0] for intersection in intersections]
        y_array = [intersection[1] for intersection in intersections]
        return np.median(x_array), np.median(y_array)
    else:
        return int(img_shape[1] // 2), int(img_shape[0] // 2)


def save_results(img, v_point, name):
    result = img.copy()

    try:
        os.mkdir(f"../res/{name.split('_')[0]}/")
    except FileExistsError:
        pass

    cv2.circle(result, (int(v_point[0]), int(v_point[1])), radius=5, color=(255, 0, 255), thickness=-1)
    cv2.imwrite(f'../res/{name}/result.jpg', result)


def run(file_name, sigma, save=False):
    img = cv2.imread(file_name)

    name = file_name.split('/')[-1].split('.')[0] + time.strftime("_%H_%M")

    lines = get_lines(img, sigma)
    intersections = find_intersections(lines)
    v_point = find_vp(intersections, img.shape)

    if save:
        save_results(img, v_point, name)

    return v_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data')
    parser.add_argument('--ans', help='path to file with answers')
    args = parser.parse_args()

    path2data = args.data
    path2res = args.ans

    p_sigma = 0.2

    res_dict = {}

    for f in tqdm(os.listdir(path2data)):
        try:
            res_dict[f] = run(os.path.join(path2data, f'{f}'),
                              p_sigma)
        except Exception as e:
            print(f)
            print(e)

    write_to_json(path2res, res_dict)
