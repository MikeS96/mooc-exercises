from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape # Shape of image
    w_half = w // 2 # half-width of image
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    ##### Attempt 1 ##### (Best attempt)
    res[220:480, w_half:] = np.tile(np.arange(w_half + 100, 100, step = -1), 
                                                           (480 - 220, 1)) * (-0.1 / (w_half + 100))
    res[220:480, :w_half] = np.tile(np.arange(100, 100 + w_half, step = 1), 
                                                           (480 - 220, 1)) * (0.1 / (w_half + 100))
    #####################
    ##### Attempt 2 #####
    # res[100:200, w_half:] = 0.5
    # res[200:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                       (480 - 200, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 3 #####
    # res[100:150, w_half:] = np.tile(np.arange(50, 0, step = -1), (320, 1)).T * (1 / 50)
    # res[150:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                       (480 - 150, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 4 #####
    # res[100:200, w_half:] = 0 #np.tile(np.arange(100, 0, step = -1), (320, 1)).T * (1 / 100)
    # res[200:480, w_half:] = np.tile(np.arange(w_half + 300, 300, step = -1), 
    #                                                       (480 - 200, 1)) * (-1 / (w_half + 300))
    #####################
    ##### Attempt 5 #####
    # res[240:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                        (480 - 240, 1)) * (-1 / w_half)
    # res[240:480, :w_half] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                        (480 - 240, 1)) * (0.5 / w_half)
    #####################
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape # Shape of image
    w_half = w // 2 # half-width of image
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    ##### Attempt 1 ##### (Best attempt)
    res[220:480, :w_half] = np.tile(np.arange(100, 100 + w_half, step = 1), 
                                                           (480 - 220, 1)) * (-0.1 / (w_half + 100))
    res[220:480, w_half:] = np.tile(np.arange(100 + w_half, 100, step = -1), 
                                                           (480 - 220, 1)) * (0.1 / (w_half + 100))
    #####################
    ##### Attempt 2 #####
    # res[100:200, 0:w_half] = 0.5
    # res[200:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                        (480 - 200, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 3 #####
    # res[100:150, 0:w_half] = np.tile(np.arange(50, 0, step = -1), (320, 1)).T * (1 / 50)
    # res[150:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                        (480 - 150, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 4 #####
    # res[100:200, 0:w_half] = 0 # np.tile(np.arange(100, 0, step = -1), (320, 1)).T * (1 / 100)
    # res[200:480, 0:w_half] = np.tile(np.arange(300, 300 + w_half, step = 1), 
    #                                                        (480 - 200, 1)) * (-1 / (300 + w_half))
    #####################
    ##### Attempt 5 #####
    # res[240:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                         (480 - 240, 1)) * (-1 / w_half)
    # res[240:480, w_half:] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                         (480 - 240, 1)) * (0.5 / w_half)
    #####################
    return res
