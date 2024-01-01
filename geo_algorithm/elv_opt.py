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
        trans: Tuple[int, int] = (
            np.random.randint(2, floor_num + 1),
            np.random.randint(2, floor_num + 1),
        )

    return trans


floor_num = 5
test_num = 100
trans_data: List[Tuple[int, int]] = []
prob = [
    0.7,
    0.2,
    0.1,
]  # prob[0]: 2~X階から1階に向かう人の割合, prob[1]: 1階から2~X階に向かう人の割合, prob[2]: 2~X階から2~X階に向かう人の割合

for i in range(test_num):
    trans = create_test_data(floor_num, prob)
    trans_data.append(trans)
print(trans_data)
