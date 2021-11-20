#!/usr/bin/env python
# coding: utf-8

# In[1]:


def DT_TOKEN():
    # todo change this to your duckietown token
    dt_token = "dt1-3nT8KSoxVh4MgtojEJRTEDsoPA6ME5UYkiRLxjEkeJcw556-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfQzey3ztD3t6hgL93hE7ea1zJtTwwVFX2S"
    return dt_token

def MODEL_NAME():
    # todo change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5"
    return "yolov5"


# In[2]:


def NUMBER_FRAMES_SKIPPED():
    # todo change this number to drop more frames
    # (must be a positive integer)
    # Simulation
    # return 3
    # Real
    return 1


# In[3]:


# `class` is the class of a prediction
def filter_by_classes(clas):
    # Right now, this returns True for every object's class
    # Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    detection = False
    if clas == 0:
        detection = True
    return detection


# In[4]:


# `scor` is the confidence score of a prediction
def filter_by_scores(scor):
    # Right now, this returns True for every object's confidence
    # Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return scor > 0.2


# In[5]:


# `bbox` is the bounding box of a prediction, in xyxy format
# So it is of the shape (leftmost x pixel, topmost y pixel, rightmost x pixel, bottommost y pixel)
def filter_by_bboxes(bbox):
    # Like in the other cases, return False if the bbox should not be considered.
    ### Simulation setup ###
    # detection = False
    # x1, y1, x2, y2 = bbox
    # w = x2 - x1
    # h = y2 - y1
    # area = w * h
    # total_area = 416 * 416
    # x_center = x1 + (w / 2)
    # y_center = y1 + (h / 2)
    # if x_center >= (416 / 6) and x_center <= (416 * 5 / 6):
    #     if area / total_area > 0.06:
    #         detection = True
    # else:
    #     detection = False
    ### Real-world setup ###
    detection = False
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    total_area = 416 * 416
    x_center = x1 + (w / 2)
    y_center = y1 + (h / 2)
    if x_center >= (416 / 6) and x_center <= (416 * 5 / 6):
        if area / total_area > 0.03:
            detection = True
    else:
        detection = False
    ### Debugging ###
    # print('bbox', bbox)
    # print('w', w)
    # print('h', h)
    # print('area', area)
    # print('total_area', total_area)
    # print('x_center', x_center)
    # print('y_center', y_center)
    # print('quart image', 416 / 6)
    # print('3/2 quart image', 416 * 5 / 6)
    # print('area / total_area', area / total_area)
    return detection

