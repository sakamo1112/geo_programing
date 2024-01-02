from typing import List, Tuple

import cv2
import numpy as np


def create_test_data(floor_num, prob):
    random_num = np.random.rand()  # 0~1の乱数を生成
    if random_num < prob[0]:
        trans: Tuple[int, int] = (np.random.randint(2, floor_num + 1), 1)
    elif random_num < prob[0] + prob[1]:
        trans: Tuple[int, int] = (1, np.random.randint(2, floor_num + 1))
    else:
        trans_0 = np.random.randint(2, floor_num + 1)
        trans_1 = np.random.randint(2, floor_num + 1)
        while trans_0 == trans_1:
            trans_1 = np.random.randint(2, floor_num + 1)
        trans: Tuple[int, int] = (trans_0, trans_1)
    return trans

def seek_wait_minsum(trans_data, test_num):
    # Xはxiの中央値にすればいい
    for i in range(test_num):
        if i <= 20:
            continue
        else:
            #過去20回の試行の中で、始点の中央値を求める
            print(np.median(trans_data[i-21:i][0]))

    return 0


if __name__ == "__main__":
    floor_num = 20
    test_num = 100
    trans_data: List[Tuple[int, int]] = []
    trans_img = np.zeros((floor_num, test_num, 3))
    # prob[0]: 2~X階から1階に向かう人の割合, prob[1]: 1階から2~X階に向かう人の割合, prob[2]: 2~X階から2~X階に向かう人の割合
    prob = [0.7, 0.2, 0.1]

    # データ生成
    for i in range(test_num):
        trans = create_test_data(floor_num, prob)
        trans_data.append(trans)

    seek_wait_minsum(trans_data, test_num)

    # データを画像に反映
    for i in range(test_num):
        trans = trans_data[i]
        white_fl = [max(trans), min(trans)]
        for fl_j in range(white_fl[0] - white_fl[1]+1):
            trans_img[floor_num-(white_fl[0] - fl_j)][i] = [255, 255, 255]
        trans_img[floor_num-trans[0]][i] = [0, 255, 0] # 始点: 緑
        trans_img[floor_num-trans[1]][i] = [255, 0, 230] # 終点: ピンク

    cv2.imwrite("trans.png", trans_img)
