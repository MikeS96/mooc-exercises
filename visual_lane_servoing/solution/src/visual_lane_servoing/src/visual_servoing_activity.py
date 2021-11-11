#!/usr/bin/env python
# coding: utf-8

# In[24]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    h, w = shape
    w_half = w // 2
    steer_matrix_left_lane = np.zeros(shape=shape, dtype="float32")
    ### Simulation ###
    # steer_matrix_left_lane[int(h * 5 / 8):, :w_half] = -0.0225
    ### Real robot ###
    steer_matrix_left_lane[int(h * 5 / 8):, :w_half] = -0.01

    return steer_matrix_left_lane


# In[25]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    h, w = shape
    w_half = w // 2
    steer_matrix_right_lane = np.zeros(shape=shape, dtype="float32")
    ### Simulation ###
    # steer_matrix_right_lane[int(h * 5 / 8):, w_half:] = 0.01 
    ### Real robot ###
    steer_matrix_right_lane[int(h * 5 / 8):, w_half:] = 0.01

    return steer_matrix_right_lane


# In[26]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    
    h, w, _ = image.shape
    
    ####### Transforming image #######
    # OpenCV uses BGR by default, whereas matplotlib uses RGB, so we generate an RGB version for the sake of visualization
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ####### Blurring and gradient #######
    sigma = 2.5

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
        # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)
    
    ####### Thresholding magnitude of gradient #######
    threshold = 50 
    mask_mag = (Gmag > threshold)
    
    ####### Filtering white and yellow #####
    ### Real robot ###
    white_lower_hsv = np.array([0, 0, 80])  # np.array([0, 0, 120]) 
    white_upper_hsv = np.array([180, 75, 255]) # np.array([180, 50, 255])
    yellow_lower_hsv = np.array([12, 80, 100])   
    yellow_upper_hsv = np.array([37, 255, 255]) 
    ### Simulation ###
    # white_lower_hsv = np.array([0, 0, 100])
    # white_upper_hsv = np.array([180, 50, 255])
    # yellow_lower_hsv = np.array([15, 80, 100])
    # yellow_upper_hsv = np.array([37, 255, 255]) 

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    ####### Edge based masking #######
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0
    
    ####### Masking gradient #######
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    
    ####### Final edge masking #######
    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return (mask_left_edge, mask_right_edge)

