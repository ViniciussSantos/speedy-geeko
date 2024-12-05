from typing import List
import cv2
import numpy as np


def get_input_dimensions(env):
    x, y = env.observation_space.shape
    return int(x / 8), int(y / 8)


def generate_input_data(observale_env, imgarray: list, x: int, y: int):
    imgarray.clear()
    observale_env = cv2.resize(observale_env, (x, y))
    observale_env = cv2.cvtColor(observale_env, cv2.COLOR_BGR2GRAY)
    observale_env = np.reshape(observale_env, (x, y))

    for i in observale_env:
        for j in i:
            imgarray.append(j)

    return imgarray


def close_msg(ram: List):
    if ram[0x1426] != 0:
        close = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        [env.step(close) for _ in range(2)]


def score_x_pos(x, max_x, current_fit, count):
    if x > max_x:
        current_fit += x / 100
        max_x = x
        count = 0
        return max_x, current_fit, count

    count += 1
    current_fit -= 10
    return max_x, current_fit, count


def score_y_pos(y, max_y, current_fit):
    if y < max_y:
        current_fit += 10
        max_y = y
    return max_y, current_fit


def score_collected_coins(coins, max_coins, current_fit):
    if coins > 0 and coins > max_coins:
        current_fit += coins - max_coins
        max_coins = max 
    return max_coins, current_fit
def score_collected_yoshicoins(yoshicoins, max_yoshicoins, current_fit):
    if yoshicoins > 0 and yoshicoins > max_yoshicoins:
        current_fit += (yoshicoins - max_yoshicoins) * 10
        max_yoshicoins = max 
    return max_yoshicoins, current_fit
