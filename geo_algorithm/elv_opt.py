import copy
from typing import List, Tuple

import cv2
import numpy as np


def create_test_data(floor_num: int, prob: List[float]) -> Tuple[int, int, str]:
    """
    This function creates test data for the elevator optimization algorithm.

    Parameters:
    floor_num (int): The number of floors in the building.
    prob (List[float]): A list of probabilities for each floor.

    Returns:
    Tuple[int, int, str]: A tuple that represents the start and end floors for the elevator trip and the type of the user.
    """
    random_num = np.random.rand()  # 0~1の乱数を生成
    if random_num < prob[0]:
        trans = (np.random.randint(2, floor_num + 1), 1, "user")
    elif random_num < prob[0] + prob[1]:
        trans = (1, np.random.randint(2, floor_num + 1), "user")
    else:
        trans_0 = np.random.randint(2, floor_num + 1)
        trans_1 = np.random.randint(2, floor_num + 1)
        while trans_0 == trans_1:
            trans_1 = np.random.randint(2, floor_num + 1)
        trans = (trans_0, trans_1, "user")
    return trans


def create_wait_data(
    trans_data: List[Tuple[int, int, str]], trans_img: np.ndarray
) -> Tuple[List[Tuple[int, int, str]], np.ndarray]:
    """
    This function creates waiting data for the elevator optimization algorithm.

    Parameters:
    trans_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the elevator trip and the type of the user.
    trans_img (np.ndarray): An array representing the transition image.

    Returns:
    Tuple[List[Tuple[int, int, str]], np.ndarray]: A tuple containing the updated list of tuples representing the start and end floors for the elevator trip and the type of the user, and the updated transition image.
    """
    """elv_move_img = np.copy(trans_img)
    elv_move_data = copy.copy(trans_data)"""
    elv_move_img = copy.deepcopy(trans_img)
    elv_move_data = copy.deepcopy(trans_data)
    pls_count2 = 0
    for i in range(len(elv_move_data)):
        if i >= len(elv_move_data) - 1:
            # 最後のデータは処理しない
            continue
        else:
            if elv_move_data[i + pls_count2 - 1][1] == elv_move_data[i + pls_count2][0]:
                continue
            else:
                start = elv_move_data[i + pls_count2 - 1][1]
                end = elv_move_data[i + pls_count2][0]
                elv_move_data.insert(i + pls_count2, (start, end, "toride"))
                elv_move_img = np.insert(
                    elv_move_img, i + pls_count2, [0, 0, 0], axis=1
                )
                white_fl = [max(start, end), min(start, end)]
                for fl_j in range(white_fl[0] - white_fl[1] + 1):
                    elv_move_img[floor_num - (white_fl[0] - fl_j)][i + pls_count2] = [
                        0,
                        0,
                        255,
                    ]
                elv_move_img[floor_num - start][i + pls_count2] = [0, 255, 0]  # 始点: 緑
                elv_move_img[floor_num - end][i + pls_count2] = [
                    110,
                    200,
                    250,
                ]  # 終点: 黄色
                pls_count2 += 1

    return elv_move_data, elv_move_img


def create_mid_stop_data(
    trans_data: List[Tuple[int, int, str]],
    trans_img: np.ndarray,
    stop_fl_list: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int, str]], np.ndarray]:
    """
    This function creates mid-stop data for the elevator optimization algorithm.

    Parameters:
    trans_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the elevator trip and the type of the user.
    trans_img (np.ndarray): An array representing the transition image.
    stop_fl_list (List[Tuple[int, int]]): A list of tuples representing the stop floors.

    Returns:
    Tuple[List[Tuple[int, int, str]], np.ndarray]: A tuple containing the updated list of tuples representing the start and end floors for the elevator trip and the type of the user, and the updated transition image.
    """
    pls_count = 1  # idx=0を除いているので1から始める
    for i, stop in enumerate(stop_fl_list):
        if trans_data[i + pls_count - 1][1] == stop[1]:
            continue
        else:
            start = trans_data[i + pls_count - 1][1]
            end = stop[1]
            white_fl = [max(start, end), min(start, end)]
            trans_data.insert(i + pls_count, (start, end, "elv"))
            trans_img = np.insert(trans_img, i + pls_count, [0, 0, 0], axis=1)
            for fl_j in range(white_fl[0] - white_fl[1] + 1):
                trans_img[floor_num - (white_fl[0] - fl_j)][i + pls_count] = [
                    255,
                    150,
                    100,
                ]
            trans_img[floor_num - start][i + pls_count] = [0, 255, 0]  # 始点: 緑
            trans_img[floor_num - end][i + pls_count] = [255, 0, 0]  # 終点: 青
            pls_count += 1

    return trans_data, trans_img


def seek_wait_minsum(
    trans_data: List[Tuple[int, int, str]], trans_img: np.ndarray, test_num: int
) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """
    This function seeks the minimum sum of wait times for the elevator optimization algorithm.

    Parameters:
    trans_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the elevator trip and the type of the user.
    trans_img (np.ndarray): An array representing the transition image.
    test_num (int): The number of tests to run.

    Returns:
    Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]: A tuple containing the normal elevator data and the optimized elevator data.
    """
    trans_data = copy.deepcopy(trans_data)
    # 絶対偏差和を最小化したいとき、Xはxiの中央値にすればいい
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
    elv_normal_data, elv_normal_img = create_wait_data(trans_data, trans_img)
    cv2.imwrite("normal.png", elv_normal_img)

    # 中継の停車階までの移動を画像に反映(停車階は番号の前に挿入するind:11なら10の位置に挿入)
    trans_data1, trans_img1 = create_mid_stop_data(trans_data, trans_img, stop_fl_list)

    # 中継の停車階から次の始点までの移動を画像に反映(待ちの移動を赤色にする)
    elv_move_minsum_data, elv_move_minsum_img = create_wait_data(
        trans_data1, trans_img1
    )
    cv2.imwrite("minsum.png", elv_move_minsum_img)

    return elv_normal_data, elv_move_minsum_data


def seek_DevSumOfSquares_minimize(
    trans_data: List[Tuple[int, int, str]], trans_img: np.ndarray, test_num: int
) -> List[Tuple[int, int, str]]:
    """
    This function seeks to minimize the sum of squares of deviations for the elevator optimization algorithm.

    Parameters:
    trans_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the elevator trip and the type of the user.
    trans_img (np.ndarray): An array representing the transition image.
    test_num (int): The number of tests to run.

    Returns:
    List[Tuple[int, int, str]]: A list of tuples representing the optimized elevator data.
    """
    trans_data = copy.deepcopy(trans_data)
    # 偏差平方和を最大化したいとき、Xはxiの平均にすればいい
    stop_fl_list: List[Tuple[int, int]] = []
    for i in range(test_num):
        if i == 0:
            continue
        elif i <= 20:
            stop_fl = int(trans_data[i - 1][1])
            stop_fl_list.append((i, stop_fl))
        else:
            user_data = [data for data in trans_data[:i] if data[2] == "user"]
            stop_fl = int(np.mean([data[0] for data in user_data[-20:]]))
            stop_fl_list.append((i, stop_fl))

    # 中継の停車階までの移動を画像に反映(停車階は番号の前に挿入するind:11なら10の位置に挿入)
    trans_data2, trans_img2 = create_mid_stop_data(trans_data, trans_img, stop_fl_list)

    # 中継の停車階から次の始点までの移動を画像に反映(待ちの移動を赤色にする)
    elv_move_mindev_data, elv_move_mindev_img = create_wait_data(
        trans_data2, trans_img2
    )
    cv2.imwrite("mindev.png", elv_move_mindev_img)

    return elv_move_mindev_data


def calculate_sum_of_wait(elv_data: List[Tuple[int, int, str]]) -> Tuple[float, int]:
    """
    This function calculates the sum of wait times and the count of 'toride' in the elevator data.

    Parameters:
    elv_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the elevator trip and the type of the user.

    Returns:
    Tuple[float, int]: A tuple containing the average wait time and the count of 'toride'.
    """
    wait_time = 0
    toride_counter = 0
    for i in range(len(elv_data)):
        if i < 20:
            continue
        else:
            if elv_data[i][2] == "toride":
                wait_time += abs(elv_data[i][0] - elv_data[i][1])
                toride_counter += 1
    ave_wait = round(wait_time / toride_counter, 2)

    return ave_wait, toride_counter


def find_average_wait_time(
    elv_normal_data: List[Tuple[int, int, str]],
    elv_move_minsum_data: List[Tuple[int, int, str]],
    elv_move_mindev_data: List[Tuple[int, int, str]],
) -> Tuple[List[float], List[int]]:
    """
    This function calculates the average wait time for the normal and optimized elevator data.

    Parameters:
    elv_normal_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the normal elevator trip and the type of the user.
    elv_move_minsum_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the optimized elevator trip and the type of the user.

    Returns:
    Tuple[List[float], List[int]]: A tuple containing the average wait times for the normal and optimized elevator data, and the count of 'toride' for the normal and optimized elevator data.
    """
    ave_wait_normal, toride_counter_normal = calculate_sum_of_wait(elv_normal_data)
    ave_wait_minsum, toride_counter_minsum = calculate_sum_of_wait(elv_move_minsum_data)
    ave_wait_mindev, toride_counter_mindev = calculate_sum_of_wait(elv_move_mindev_data)

    wait_ave_time = [ave_wait_normal, ave_wait_minsum, ave_wait_mindev]
    toride_counter = [
        toride_counter_normal,
        toride_counter_minsum,
        toride_counter_mindev,
    ]

    return wait_ave_time, toride_counter


def calculate_square_sum_of_wait(
    elv_data: List[Tuple[int, int, str]], wait_ave_time: float, toride_counter: int
) -> float:
    var_wait: float = 0
    for i in range(len(elv_data)):
        if i < 20:
            continue
        else:
            if elv_data[i][2] == "toride":
                var_wait += (abs(elv_data[i][0] - elv_data[i][1]) - wait_ave_time) ** 2
    var_wait = round(var_wait / toride_counter, 2)
    return var_wait


def find_wait_time_variance(
    elv_normal_data: List[Tuple[int, int, str]],
    elv_move_minsum_data: List[Tuple[int, int, str]],
    elv_move_mindev_data: List[Tuple[int, int, str]],
    wait_ave_time: List[float],
    toride_counter: List[int],
) -> List[float]:
    """
    This function calculates the variance of the wait time for the normal and optimized elevator data.

    Parameters:
    elv_normal_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the normal elevator trip and the type of the user.
    elv_move_minsum_data (List[Tuple[int, int, str]]): A list of tuples representing the start and end floors for the optimized elevator trip and the type of the user.
    wait_ave_time (List[float]): A list of average wait times for the normal and optimized elevator data.
    toride_counter (List[int]): A list of counts of 'toride' for the normal and optimized elevator data.

    Returns:
    List[float]: A list of variances of the wait times for the normal and optimized elevator data.
    """
    var_wait_normal = calculate_square_sum_of_wait(
        elv_normal_data, wait_ave_time[0], toride_counter[0]
    )
    var_wait_minsum = calculate_square_sum_of_wait(
        elv_move_minsum_data, wait_ave_time[1], toride_counter[1]
    )
    var_wait_mindev = calculate_square_sum_of_wait(
        elv_move_mindev_data, wait_ave_time[2], toride_counter[2]
    )
    wait_time_var = [var_wait_normal, var_wait_minsum, var_wait_mindev]

    return wait_time_var


if __name__ == "__main__":
    floor_num = 20
    test_num = 1000
    trans_userdata = []
    trans_userimg = np.zeros((floor_num, test_num, 3))
    # prob[0]: 2~X階から1階に向かう人の割合, prob[1]: 1階から2~X階に向かう人の割合, prob[2]: 2~X階から2~X階に向かう人の割合
    prob = [0.7, 0.2, 0.1]

    # データ生成
    for i in range(test_num):
        trans = create_test_data(floor_num, prob)
        trans_userdata.append(trans)

    # 生成データを画像に反映
    for i in range(test_num):
        trans = trans_userdata[i]
        white_fl = [max(trans[:2]), min(trans[:2])]
        for fl_j in range(white_fl[0] - white_fl[1] + 1):
            trans_userimg[floor_num - (white_fl[0] - fl_j)][i] = [255, 255, 255]
        trans_userimg[floor_num - trans[0]][i] = [0, 255, 0]  # 始点: 緑
        trans_userimg[floor_num - trans[1]][i] = [255, 0, 230]  # 終点: ピンク
    cv2.imwrite("user_trans.png", trans_userimg)

    elv_normal_data, elv_move_minsum_data = seek_wait_minsum(
        trans_userdata, trans_userimg, test_num
    )
    elv_move_mindev_data = seek_DevSumOfSquares_minimize(
        trans_userdata, trans_userimg, test_num
    )

    # 待ち時間の平均・分散を求める
    wait_ave_time, toride_counter = find_average_wait_time(
        elv_normal_data, elv_move_minsum_data, elv_move_mindev_data
    )
    wait_time_var = find_wait_time_variance(
        elv_normal_data,
        elv_move_minsum_data,
        elv_move_mindev_data,
        wait_ave_time,
        toride_counter,
    )

    print("-------------------------------------------------------------")
    print("     挙動     | 平均待ち時間 [階分] |          分散            ")
    print("-------------------------------------------------------------")
    print(
        "    通常      |    ", wait_ave_time[0], "           |        ", wait_time_var[0]
    )
    print(
        " MinSum挙動   |    ", wait_ave_time[1], "           |        ", wait_time_var[1]
    )
    print(
        " MinDev挙動   |    ", wait_ave_time[2], "           |        ", wait_time_var[2]
    )
    print("-------------------------------------------------------------")
