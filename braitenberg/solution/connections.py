from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape # Shape of image
    w_half = w // 2 # half-width of image
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    ##### Attempt 1 #####
    # res[100:150, 0:w_half] = -0.5 # Lover part
    # res[150:480, w_half:int(w_half + w_half*2/3)] = -1 
    # res[150:480, int(w_half + w_half*2/3):-1] = 0.2 
    #####################
    ##### Attempt 2 ##### (Best attempt)
    # res[100:150, w_half:] = 0.0 # Agression part 0.5
    # res[150:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                        (480 - 150, 1)) * (-1.0 / w_half)
    # res[240:480, w_half:] = np.tile(np.arange(w_half + 100, 100, step = -1), 
    #                                                        (480 - 240, 1)) * (-0.1 / (w_half + 100))
    # res[240:480, :w_half] = np.tile(np.arange(100, w_half + 100, step = 1), 
    #                                                        (480 - 240, 1)) * (0.1 / (w_half + 100))
    # res[200:480, w_half:] = -0.5
    # res[240:480, :w_half] = 1e-4
    res[240:480, w_half:] = np.tile(np.arange(w_half + 100, 100, step = -1), 
                                                           (480 - 240, 1)) * (-0.2 / (w_half + 100))
    # res[240:480, :w_half] = np.tile(np.arange(100, w_half + 100, step = 1), 
    #                                                        (480 - 240, 1)) * (0.1 / (w_half + 100))
    #####################
    ##### Attempt 3 ##### (Works incredible well)
    # res[100:200, w_half:] = 0.5
    # res[200:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                       (480 - 200, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 4 ##### (Works incredible well too :) - Similar to attemp 1
    # res[100:150, w_half:] = np.tile(np.arange(50, 0, step = -1), (320, 1)).T * (1 / 50)
    # res[150:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                       (480 - 150, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 5 ##### (The worst but not really bad)
    # res[100:200, w_half:] = 0 #np.tile(np.arange(100, 0, step = -1), (320, 1)).T * (1 / 100)
    # res[200:480, w_half:] = np.tile(np.arange(w_half + 300, 300, step = -1), 
    #                                                       (480 - 200, 1)) * (-1 / (w_half + 300))
    #####################
    ##### Attempt 6 ##### So so
    # res[240:480, w_half:] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                        (480 - 240, 1)) * (-1 / w_half)
    # res[240:480, :w_half] = np.tile(np.arange(w_half, 0, step = -1), 
    #                                                        (480 - 240, 1)) * (0.5 / w_half)
    #####################
    ##### Attempt house #####
    # res[:, w_half:] = np.tile(np.arange(0, h, step = 1), (w_half, 1)).T * (-1.0 / h)
    # res[:, :w_half] = np.tile(np.arange(0, h, step = 1), (w_half, 1)).T * (1.0 / h)
    #res[199:, w_half:] = np.tile(np.arange(200, h - 199 + 200, step = 1), (w_half, 1)).T * (-1.0 / (h - 199 + 200))
    #####################
    ##### Attempt lab ##### (Best attempt)
    # res[100:150, 0:w_half] = 0.0 # Agression part 0.5
    # res[150:480, w_half:] = -1
    #####################
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape # Shape of image
    w_half = w // 2 # half-width of image
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    ##### Attempt 1 #####
    # res[100:150, w_half:-1] = -0.5 # Lover part 0.8
    # res[150:480, int(w_half*1/3):w_half] = -1
    # res[150:480, :int(w_half*1/3)] = 0.2 # 0.2
    #####################
    ##### Attempt 2 ##### (Best attempt)
    # res[100:150, 0:w_half] = 0.0 # Agression part
    # res[150:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                         (480 - 150, 1)) * (-1.0 / w_half)
    # res[240:480, :w_half] = np.tile(np.arange(100, 100 + w_half, step = 1), 
    #                                                        (480 - 240, 1)) * (-0.2 / (w_half + 100))
    # res[240:480, w_half:] = np.tile(np.arange(w_half + 100, 100, step = -1), 
    #                                                        (480 - 240, 1)) * (0.2 / (w_half + 100))
    # res[200:480, :w_half] = -0.5
    # res[240:480, w_half:] = 1e-4
    res[240:480, :w_half] = np.tile(np.arange(100, 100 + w_half, step = 1), 
                                                           (480 - 240, 1)) * (-0.2 / (w_half + 100))
    # res[240:480, w_half:] = np.tile(np.arange(100 + w_half, 100, step = -1), 
    #                                                        (480 - 240, 1)) * (0.2 / (w_half + 100))
    #####################
    ##### Attempt 3 ##### (Works incredible well)
    # res[100:200, 0:w_half] = 0.5
    # res[200:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                        (480 - 200, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 4 ##### (Works incredible well too :) - Similar to attemp 1
    # res[100:150, 0:w_half] = np.tile(np.arange(50, 0, step = -1), (320, 1)).T * (1 / 50)
    # res[150:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                        (480 - 150, 1)) * (-1 / w_half)
    #####################
    ##### Attempt 5 ##### (The worst but not really bad)
    # res[100:200, 0:w_half] = 0 # np.tile(np.arange(100, 0, step = -1), (320, 1)).T * (1 / 100)
    # res[200:480, 0:w_half] = np.tile(np.arange(300, 300 + w_half, step = 1), 
    #                                                        (480 - 200, 1)) * (-1 / (300 + w_half))
    #####################
    ##### Attempt 6 ##### So so
    # res[240:480, 0:w_half] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                         (480 - 240, 1)) * (-1 / w_half)
    # res[240:480, w_half:] = np.tile(np.arange(0, w_half, step = 1), 
    #                                                         (480 - 240, 1)) * (0.5 / w_half)
    #####################
    ##### Attempt house ##### 
    # res[:, 0:w_half] = np.tile(np.arange(0, h, step = 1), (w_half, 1)).T * (-1.0 / h)
    # res[:, w_half:] = np.tile(np.arange(0, h, step = 1), (w_half, 1)).T * (1.0 / h)
    #res[199:, 0:w_half] = np.tile(np.arange(200, h - 199 + 200, step = 1), (w_half, 1)).T * (-1.0 / (h - 199 + 200))
    #####################
    ##### Attempt lab ##### (Best attempt)
    # res[100:150, w_half:] = 0.0 # Agression part
    # res[150:480, 0:w_half] = -1
    #####################
    return res
