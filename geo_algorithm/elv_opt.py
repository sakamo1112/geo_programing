from typing import List, Tuple

import cv2
import numpy as np
import copy


def create_test_data(floor_num: int, prob: List[float]) -> Tuple[int, int]:
    random_num = np.random.rand()  # 0~1の乱数を生成
    if random_num < prob[0]:
        trans: Tuple[int, int] = (np.random.randint(2, floor_num + 1), 1, "user")
    elif random_num < prob[0] + prob[1]:
        trans: Tuple[int, int] = (1, np.random.randint(2, floor_num + 1), "user")
    else:
        trans_0 = np.random.randint(2, floor_num + 1)
        trans_1 = np.random.randint(2, floor_num + 1)
        while trans_0 == trans_1:
            trans_1 = np.random.randint(2, floor_num + 1)
        trans: Tuple[int, int] = (trans_0, trans_1, "user")
    return trans

def create_wait_data(trans_data, trans_img):
    elv_move_img = np.copy(trans_img)
    elv_move_data = copy.copy(trans_data)
    pls_count2 = 0
    for i in range(len(trans_data)):
        if i >= len(trans_data)-1:
            # 最後のデータは処理しない
            continue
        else:
            if elv_move_data[i+pls_count2-1][1] == elv_move_data[i+pls_count2][0]:
                continue
            else:
                start = elv_move_data[i+pls_count2-1][1]
                end = elv_move_data[i+pls_count2][0]
                elv_move_data.insert(i+pls_count2, (start, end, "toride"))
                elv_move_img = np.insert(elv_move_img, i+pls_count2, [0, 0, 0], axis=1)
                white_fl = [max(start, end), min(start, end)]
                for fl_j in range(white_fl[0] - white_fl[1] + 1):
                    elv_move_img[floor_num - (white_fl[0] - fl_j)][i+pls_count2] = [0, 0, 255]
                elv_move_img[floor_num - start][i+pls_count2] = [0, 255, 0]  # 始点: 緑
                elv_move_img[floor_num - end][i+pls_count2] = [110, 200, 250]  # 終点: 黄色
                pls_count2 += 1
        
    return elv_move_data, elv_move_img

def create_mid_stop_data(trans_data, trans_img, stop_fl_list):
    pls_count = 1 # idx=0を除いているので1から始める
    for i, stop in enumerate(stop_fl_list):
        if trans_data[i+pls_count-1][1] == stop[1]:
            continue
        else:
            start = trans_data[i+pls_count-1][1]
            end = stop[1]
            white_fl = [max(start, end), min(start, end)]
            trans_data.insert(i+pls_count, (start, end, "elv"))
            trans_img = np.insert(trans_img, i+pls_count, [0, 0, 0], axis=1)
            for fl_j in range(white_fl[0] - white_fl[1] + 1):
                trans_img[floor_num - (white_fl[0] - fl_j)][i+pls_count] = [255, 255, 255]
            trans_img[floor_num - start][i+pls_count] = [0, 255, 0]  # 始点: 緑
            trans_img[floor_num - end][i+pls_count] = [255, 0, 0]  # 終点: 青
            pls_count += 1

    return trans_data, trans_img


def seek_wait_minsum(trans_img, floor_num, trans_data, test_num: int):
    # Xはxiの中央値にすればいい
    stop_fl_list: List[Tuple[int, int]] = []
    for i in range(test_num):
        if i == 0:
            continue
        elif i <= 20:
            stop_fl = int(trans_data[i - 1][1])
            stop_fl_list.append((i, stop_fl))
        else:
            user_data = [data for data in trans_data[:i] if data[2] == "user"]
            stop_fl = int(np.median([data[0] for data in user_data[-20:]]))
            stop_fl_list.append((i, stop_fl))
    elv_move_nominsum_data, elv_move_nominsum_img = create_wait_data(trans_data, trans_img)
    cv2.imwrite("nominsum.png", elv_move_nominsum_img)
            
    # 中継の停車階までの移動を画像に反映(停車階は番号の前に挿入するind:11なら10の位置に挿入)
    trans_data, trans_img = create_mid_stop_data(trans_data, trans_img, stop_fl_list)

    # 中継の停車階から次の始点までの移動を画像に反映(待ちの移動を赤色にする)
    elv_move_minsum_data, elv_move_minsum_img = create_wait_data(trans_data, trans_img)
    cv2.imwrite("minsum.png", elv_move_minsum_img)

    # 待ち時間の平均を求める

    # 待ち時間の分散を求める

    return elv_move_minsum_data

def seek_0wait_maximize():
    pass


if __name__ == "__main__":
    floor_num = 20
    test_num = 1000
    trans_data: List[Tuple[int, int]] = []
    trans_img = np.zeros((floor_num, test_num, 3))
    # prob[0]: 2~X階から1階に向かう人の割合, prob[1]: 1階から2~X階に向かう人の割合, prob[2]: 2~X階から2~X階に向かう人の割合
    prob = [0.7, 0.2, 0.1]

    # データ生成
    for i in range(test_num):
        trans = create_test_data(floor_num, prob)
        trans_data.append(trans)

    # seek_wait_minsum(trans_img, floor_num, trans_data, test_num)

    # データを画像に反映
    for i in range(test_num):
        trans = trans_data[i]
        white_fl = [max(trans[:2]), min(trans[:2])]
        for fl_j in range(white_fl[0] - white_fl[1] + 1):
            trans_img[floor_num - (white_fl[0] - fl_j)][i] = [255, 255, 255]
        trans_img[floor_num - trans[0]][i] = [0, 255, 0]  # 始点: 緑
        trans_img[floor_num - trans[1]][i] = [255, 0, 230]  # 終点: ピンク
    cv2.imwrite("trans.png", trans_img)

    elv_move_data = seek_wait_minsum(trans_img, floor_num, trans_data, test_num)
