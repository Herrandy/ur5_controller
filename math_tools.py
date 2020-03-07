#!/usr/bin/env python
import numpy as np
import math

def axis_ang2rotm(angle_vec):
    """
    Converts an axis angle representation to rotation matrix (taken from tf.transformations)
    :param angle_vec: 3D rotation
    :return: 3x3 rotation matrix
    """
    assert angle_vec.size == 3

    theta = math.sqrt(angle_vec[0]**2+angle_vec[1]**2+angle_vec[2]**2)
    if theta == 0.:
        return np.identity(3, dtype=float)
    
    cs = np.cos(theta)
    si = np.sin(theta)
    e1 = angle_vec[0]/theta
    e2 = angle_vec[1]/theta
    e3 = angle_vec[2]/theta
            
    R=np.zeros((3, 3))
    R[0, 0] = (1-cs)*e1**2+cs
    R[0, 1] = (1-cs)*e1*e2-e3*si
    R[0, 2] = (1-cs)*e1*e3+e2*si
    R[1, 0] = (1-cs)*e1*e2+e3*si
    R[1, 1] = (1-cs)*e2**2+cs
    R[1, 2] = (1-cs)*e2*e3-e1*si
    R[2, 0] = (1-cs)*e1*e3-e2*si
    R[2, 1] = (1-cs)*e2*e3+e1*si
    R[2, 2] = (1-cs)*e3**2+cs
    return R


def pose_difference(target, current):
    '''
    Differenence between two 6D pose vectors
    :param target: 6D pose vector
    :param current: 6D pose vector
    :return:
    '''
    delta = (target - current)
    delta[3:] = rotation_diff(target, current)
    return delta

def pose2tf(pose):
    """
    Converts 6D pose vector 4x4 pose matrix
    :param pose: 6D pose vector
    :return: 4x4 matrix
    """
    M = np.eye(4)
    M[:3, :3] = axis_ang2rotm(pose[3:])
    M[:3, 3] = pose[:3]
    return M


def tf2pose(M):
    """
    Converts 4x4 pose matrix to 6D pose vector
    :param M: 4x4 matrix
    :return: 6D pose vector
    """
    output = np.zeros(6)
    output[:3] = M[:3, 3]
    angle, direction, _ = rotm2axis_ang(M)
    output[3:] = direction * angle
    return output


def rotm2axis_ang(M):
    """
    Converts rotation matrix to axis angle representation (taken from tf.transformations)
    :param M: 4x4 pose matrix
    :return: components for full rotation vector
    """
    R = np.array(M, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point

def rotation_diff(goal, curr):
    """
    Calculates difference between two rotation matrices
    :param goal: goal orientation
    :param curr: current orientation
    :return: rotation vector
    """
    goal_T = pose2tf(goal)[:3, :3]
    curr_T = pose2tf(curr)[:3, :3]
    diff_T = np.dot(goal_T, np.linalg.inv(curr_T))
    temp = np.eye(4)
    temp[:3, :3] = diff_T
    angle, direction, _ = rotm2axis_ang(temp)
    return direction * angle
