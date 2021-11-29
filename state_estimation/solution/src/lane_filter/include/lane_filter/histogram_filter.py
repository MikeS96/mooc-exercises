#!/usr/bin/env python
# coding: utf-8

# In[1]:


# start by importing some things we will need
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy, multivariate_normal
from math import floor, sqrt


# In[5]:


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution
def histogram_prior(belief, grid_spec, mean_0, cov_0):
        pos = np.empty(belief.shape + (2,))
        pos[:, :, 0] = grid_spec["d"]
        pos[:, :, 1] = grid_spec["phi"]
        RV = multivariate_normal(mean_0, cov_0)
        belief = RV.pdf(pos)
        return belief

# Now let's define the predict function
def compute_pose(robot_spec, grid_phi, left_encoder_ticks, right_encoder_ticks):
        """
        Compute x and v using ticks
        """
        # Obtaining parameters from the robot
        N_tot = robot_spec['encoder_resolution']
        R = robot_spec['wheel_radius']
        baseline_wheel2wheel = robot_spec['wheel_baseline']
        # Compute delta phi for left and right wheel [rads]
        alpha = 2 * np.pi / N_tot # rotation per tick in radians 
        delta_phi_left = left_encoder_ticks * alpha
        delta_phi_right = right_encoder_ticks * alpha
        # Distance travelled by each wheel [m]
        d_left = R * delta_phi_left 
        d_right = R * delta_phi_right
        # Define distance travelled by the robot, in body frame [m]
        d_A = (d_left + d_right)/2
        # Define distance travelled by the robot in Y, in world frame [m]
        Dy = d_A * np.sin(grid_phi)
        # Define rotation of the robot [rad]
        Dtheta = (d_right - d_left)/baseline_wheel2wheel

        return Dy, Dtheta

def histogram_predict(belief, dt, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
        belief_in = belief
        delta_t = dt
        
        d, theta = compute_pose(robot_spec, grid_spec['phi'], left_encoder_ticks, right_encoder_ticks)
        v = d / delta_t # replace this with a function that uses the encoder 
        w = theta / delta_t # replace this with a function that uses the encoder
        
        # TODO propagate each centroid forward using the kinematic function
        d_t = grid_spec['d'] + d # replace this with something that adds the new odometry
        phi_t = grid_spec['phi'] + theta # replace this with something that adds the new odometry

        p_belief = np.zeros(belief.shape)

        # Accumulate the mass for each cell as a result of the propagation step
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                # If belief[i,j] there was no mass to move in the first place
                if belief[i, j] > 0:
                    # Now check that the centroid of the cell wasn't propagated out of the allowable range
                    if (
                        d_t[i, j] > grid_spec['d_max']
                        or d_t[i, j] < grid_spec['d_min']
                        or phi_t[i, j] < grid_spec['phi_min']
                        or phi_t[i, j] > grid_spec['phi_max']
                    ):
                        continue
                    
                    # Now find the cell where the new mass should be added
                    i_new = np.digitize(d_t[i, j], grid_spec['d'][:, 0], right = False) - 1
                    j_new = np.digitize(phi_t[i, j], grid_spec['phi'][0, :], right = False) - 1
                    # i_new = np.int((d_t[i, j] - grid_spec['d_min']) / grid_spec['delta_d'])
                    # j_new = np.int((phi_t[i, j] - grid_spec['phi_min']) / grid_spec['delta_phi'])
                    # i_new = np.argmin(np.abs(d_t[i, j] - grid_spec["d"][:, 0]))
                    # j_new = np.argmin(np.abs(phi_t[i, j] - grid_spec["phi"][0, :]))

                    p_belief[i_new, j_new] += belief[i, j]

        # Finally we are going to add some "noise" according to the process model noise
        # This is implemented as a Gaussian blur
        s_belief = np.zeros(belief.shape)
        gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

        if np.sum(s_belief) == 0:
            return belief_in
        belief = s_belief / np.sum(s_belief)
        return belief


# In[6]:


# We will start by doing a little bit of processing on the segments to remove anything that is behing the robot (why would it be behind?)
# or a color not equal to yellow or white

def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments


# In[7]:



def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane
            d_i = -d_i
            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i


# In[8]:


def generate_measurement_likelihood(segments, road_spec, grid_spec):

    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec['d'].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if d_i > grid_spec['d_max'] or d_i < grid_spec['d_min'] or phi_i < grid_spec['phi_min'] or phi_i > grid_spec['phi_max']:
            continue

        # Find the cell index that corresponds to the measurement d_i, phi_i
        i = np.digitize(d_i, grid_spec['d'][:, 0], right = False) - 1
        j = np.digitize(phi_i, grid_spec['phi'][0, :], right = False) - 1
        # i = np.int((d_i - grid_spec['d_min']) / grid_spec['delta_d'])
        # j = np.int((phi_i - grid_spec['phi_min']) / grid_spec['delta_phi'])
        # print('i', i, 'j', j)
        
        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


# In[9]:


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)
    
    p_belief = np.zeros(belief.shape)

    if measurement_likelihood is not None:
        # Combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid probability distribution
        p_belief = belief * measurement_likelihood # replace this with something that combines the belief and the measurement_likelihood
        # Normalizing!
        if np.sum(p_belief) != 0:
            belief = p_belief / np.sum(p_belief)
        # Liam's suggestion
        else:
            belief = measurement_likelihood

    return (measurement_likelihood, belief)

