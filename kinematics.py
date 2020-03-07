#!/usr/bin/env python
import numpy as np
import math as m
from numpy.linalg import inv
from numpy import linalg as la

class KinematicsUR5:
    def __init__(self):
        self._a = [0, -0.425, -0.39225, 0, 0, 0]
        self._d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self._alpha = [m.pi / 2, 0, 0, m.pi / 2, -m.pi / 2, 0]

    def DH(self, a, alpha, d, theta):
        dh = np.array([
            [m.cos(theta), -m.sin(theta) * m.cos(alpha),  m.sin(theta) * m.sin(alpha), a * m.cos(theta)],
            [m.sin(theta),  m.cos(theta) * m.cos(alpha), -m.cos(theta) * m.sin(alpha), a * m.sin(theta)],
            [           0,                 m.sin(alpha),                m .cos(alpha),                d],
            [           0,                            0,                            0,                1]
        ])
        for i in range(np.shape(dh)[0]):
            for j in range(np.shape(dh)[1]):
                if abs(dh[i, j]) < 0.0001:
                    dh[i, j] = 0.0

        return dh

    def calculate_Jacobian(self, q):
        n_joints = len(q)
        kz = np.array([0, 0, 1])
        transformations = self.get_transformations(q)
        J_linear = np.zeros((n_joints, 3))
        J_angular = np.zeros((n_joints, 3))
        J = np.zeros((6, n_joints))

        J_angular[0] = np.dot(np.eye(3), kz)
        J_linear[0] = np.cross(J_angular[0], transformations[-1][:3, 3])
        J[0] = np.concatenate((J_linear[0], J_angular[0]), axis=0)

        for i in range(1, n_joints):
            T = transformations[i - 1]
            J_angular[i] = np.dot(T[:3, :3], kz)
            J_linear[i] = np.cross(J_angular[i], (transformations[-1][:3, 3] - T[:3, 3]))
            J[i] = np.concatenate((J_linear[i], J_angular[i]), axis=0)
        return J_linear, J_angular, J

    def get_transformations(self, joint_values):
        T = np.zeros((6, 4, 4))
        T01 = self.DH(self._a[0], self._alpha[0], self._d[0], joint_values[0])
        T12 = self.DH(self._a[1], self._alpha[1], self._d[1], joint_values[1])
        T23 = self.DH(self._a[2], self._alpha[2], self._d[2], joint_values[2])
        T34 = self.DH(self._a[3], self._alpha[3], self._d[3], joint_values[3])
        T45 = self.DH(self._a[4], self._alpha[4], self._d[4], joint_values[4])
        T56 = self.DH(self._a[5], self._alpha[5], self._d[5], joint_values[5])
        T02 = np.dot(T01, T12)
        T03 = np.dot(T02, T23)
        T04 = np.dot(T03, T34)
        T05 = np.dot(T04, T45)
        T06 = np.dot(T05, T56)
        T[0] = T01
        T[1] = T02
        T[2] = T03
        T[3] = T04
        T[4] = T05
        T[5] = T06
        return T

    def fwd_kin(self, joints):
        T01 = self.DH(self._a[0], self._alpha[0], self._d[0], joints[0])
        T12 = self.DH(self._a[1], self._alpha[1], self._d[1], joints[1])
        T23 = self.DH(self._a[2], self._alpha[2], self._d[2], joints[2])
        T34 = self.DH(self._a[3], self._alpha[3], self._d[3], joints[3])
        T45 = self.DH(self._a[4], self._alpha[4], self._d[4], joints[4])
        T56 = self.DH(self._a[5], self._alpha[5], self._d[5], joints[5])
        return np.dot(np.dot(np.dot(np.dot(np.dot(T01, T12), T23), T34), T45), T56)

    def inv_kin(self, pose):
        theta = np.zeros((6, 8))

        # theta1
        temp1 = np.array([0, 0, -self._d[5], 1])
        temp1.shape = (4, 1)
        temp2 = np.array([0, 0, 0, 1])
        temp2.shape = (4, 1)
        p05 = np.dot(pose, temp1) - temp2
        psi = m.atan2(p05[1], p05[0])
        if self._d[3] / m.sqrt(p05[1]**2 + p05[0]**2) > 1:
            phi = 0
        else:
            phi = m.acos(self._d[3] / m.sqrt(p05[1]**2 + p05[0]**2))
        theta[0, :4] = m.pi / 2 + psi + phi
        theta[0, 4:8] = m.pi / 2 + psi - phi

        # theta5
        for c in [0, 4]:
            T10 = inv(self.DH(self._a[0], self._alpha[0], self._d[0], theta[0, c]))
            T16 = np.dot(T10, pose)
            p16z = T16[2, 3]
            if (p16z - self._d[3]) / self._d[5] > 1:
                t5 = 0
            else:
                t5 = m.acos((p16z - self._d[3]) / self._d[5])
            theta[4, c:c + 1 + 1] = t5
            theta[4, c + 2:c + 3 + 1] = -t5

        # theta6
        for c in [0, 2, 4, 6]:
            T01 = self.DH(self._a[0], self._alpha[0], self._d[0], theta[0, c])
            T61 = np.dot(inv(pose), T01)
            T61zy = T61[1, 2]
            T61zx = T61[0, 2]
            t5 = theta[4, c]
            theta[5, c:c + 1 + 1] = m.atan2(-T61zy / m.sin(t5), T61zx / m.sin(t5))

        # theta3
        for c in [0, 2, 4, 6]:
            T10 = inv(self.DH(self._a[0], self._alpha[0], self._d[0], theta[0, c]))
            T65 = inv(self.DH(self._a[5], self._alpha[5], self._d[5], theta[5, c]))
            T54 = inv(self.DH(self._a[4], self._alpha[4], self._d[4], theta[4, c]))
            T14 = np.dot(np.dot(T10, pose), np.dot(T65, T54))
            temp1 = np.array([0, -self._d[3], 0, 1])
            temp1.shape = (4, 1)
            temp2 = np.array([0, 0, 0, 1])
            temp2.shape = (4, 1)
            p13 = np.dot(T14, temp1) - temp2
            p13norm2 = la.norm(p13)**2
            if (p13norm2 - self._a[1]**2 - self._a[2]**2) / (2 * self._a[1] * self._a[2]) > 1:
                t3p = 0
            else:
                t3p = m.acos((p13norm2 - self._a[1]**2 - self._a[2]**2) / (2 * self._a[1] * self._a[2]))
            theta[2, c] = t3p
            theta[2, c + 1] = -t3p

            # theta2 theta4
        for c in range(8):
            T10 = inv(self.DH(self._a[0], self._alpha[0], self._d[0], theta[0, c]))
            T65 = inv(self.DH(self._a[5], self._alpha[5], self._d[5], theta[5, c]))
            T54 = inv(self.DH(self._a[4], self._alpha[4], self._d[4], theta[4, c]))
            T14 = np.dot(np.dot(T10, pose), np.dot(T65, T54))
            temp1 = np.array([0, -self._d[3], 0, 1])
            temp1.shape = (4, 1)
            temp2 = np.array([0, 0, 0, 1])
            temp2.shape = (4, 1)
            p13 = np.dot(T14, temp1) - temp2
            p13norm = la.norm(p13)
            theta[1, c] = -m.atan2(p13[1], -p13[0]) + m.asin(self._a[2] * m.sin(theta[2, c]) / p13norm)
            T32 = inv(self.DH(self._a[2], self._alpha[2], self._d[2], theta[2, c]))
            T21 = inv(self.DH(self._a[1], self._alpha[1], self._d[1], theta[1, c]))
            T34 = np.dot(np.dot(T32, T21), T14)
            theta[3, c] = m.atan2(T34[1, 0], T34[0, 0])

        for i in range(np.shape(theta)[0]):
            for j in range(np.shape(theta)[1]):
                if theta[i, j] > m.pi:
                    theta[i, j] = theta[i, j] - 2 * m.pi
                if theta[i, j] < -m.pi:
                    theta[i, j] = theta[i, j] + 2 * m.pi

        return theta

    def get_closest_solution(self, sols_q, current_q):
        min = float('inf')
        cand_idx = 0
        for idx in range(sols_q.shape[1]):
            diff = np.linalg.norm(sols_q[:, idx] - current_q)
            if diff < min:
                min = diff
                cand_idx = idx
        return sols_q[:, cand_idx]

