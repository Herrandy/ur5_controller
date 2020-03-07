#!/usr/bin/env python
import socket
import sys
import struct
import numpy as np
import time
from threading import Thread
from math_tools import pose_difference
from kinematics import KinematicsUR5

class RTClient:
    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._robot_kin = KinematicsUR5()
        self._data = {
                      'joint_values': np.array([]),
                      'joint_velocities': np.array([]),
                      'tool_pose': np.array([]),
                      'tool_forces': np.array([]),
                      'tool_velocity': np.array([])
                      }
        self._connect(host, port)
        self._keep_alive = True
        self._data_thread = Thread(target=self._data_listener)
        self._data_thread.daemon = True
        self._data_thread.start()

    def _data_listener(self):
        while self._keep_alive:
            raw_msg = self._recv_msg(self._socket)
            if len(raw_msg) != 1104:
                return
            self._data['joint_values'] = np.array(struct.unpack('!dddddd', raw_msg[248:296]))
            self._data['joint_velocities'] = np.array(struct.unpack('!dddddd', raw_msg[296:344]))
            self._data['tool_pose'] = np.array(struct.unpack('!dddddd', raw_msg[440:488]))
            self._data['tool_velocity'] = np.array(struct.unpack('!dddddd', raw_msg[488:536]))
            self._data['tool_forces'] = np.array(struct.unpack('!dddddd', raw_msg[536:584]))

    def _connect(self, host, port):
        try:
            self._socket.connect((host, port))
        except socket.error as msg:
            print("Connection failed: %s\n" % msg)
            sys.exit(1)
        print('Connected')

    def _disconnect(self):
        self._keep_alive = False
        self._data_thread.join()
        self._socket.close()
        print('Disconnected')

    def _recv_msg(self, sock):
        raw_msglen = self._recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        return self._recvall(sock, msglen - 4)

    def _recvall(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def get_feedback(self, *options):
        if len(options) < 1:
            print("No parameters given, returning ...")
            return
        ret = []
        for key in options:
            ret.append(self._data[key])
        if len(ret) < 2:
            return ret[0]
        return ret

    def close_connection(self):
        self._disconnect()

    def move_j(self, target_q, a=0.9, v=0.9, wait=True):
        if type(target_q) is np.ndarray:
            target_q = target_q.tolist()
        assert len(target_q) == 6
        cmd = "movej({}, a={}, v={})".format(target_q, a, v)
        self._send(cmd)
        if wait:
            self._wait_joint_goal(target_q)

    def move_l(self, target_ee, a=0.9, v=0.9, wait=True):
        if type(target_ee) is np.ndarray:
            target_ee = target_ee.tolist()
        assert len(target_ee) == 6
        cmd = "movel(p{}, a={}, v={})".format(target_ee, a, v)
        self._send(cmd)
        if wait:
            self._wait_pose_goal(target_ee)

    def move_v(self, target_ee):
        current_ee = self.get_feedback('tool_pose')
        err = (target_ee - current_ee)
        while np.linalg.norm(err) % np.pi > 5.0e-3:
            current_ee, q = self.get_feedback('tool_pose', 'joint_values')
            delta = pose_difference(target_ee, current_ee)
            _, _, J = self._robot_kin.calculate_Jacobian(q)
            invJ = np.linalg.pinv(J.transpose())
            target_qd = np.dot(invJ, delta)
            cmd = "speedj({}, {}, 0.08)".format(target_qd.tolist(), 10)
            self._send(cmd)
            err = (target_ee - current_ee)

    def speed_j(self, target_qd, a=1.0, t=0.8):
        if type(target_qd) is np.ndarray:
            target_qd = target_qd.tolist()
        assert len(target_qd) == 6
        cmd = "speedj({}, {}, {})".format(target_qd, a, t)
        self._send(cmd)

    def _send(self, cmd):
        self._socket.send((cmd + "\n").encode())

    def stop(self, a=0.9):
        cmd = "stopj(a={})".format(a)
        self._socket.send((cmd + "\n").encode())

    def _wait_joint_goal(self, goal_q):
        q, qd = self.get_feedback('joint_values', 'joint_velocities')
        while np.linalg.norm(goal_q - q) > 1e-3 or np.sum(np.abs(qd)) > 1e-3:
            q, qd = self.get_feedback('joint_values', 'joint_velocities')
            time.sleep(0.01)

    def _wait_pose_goal(self, goal_ee):
        current_ee, qd = self.get_feedback('tool_pose', 'joint_velocities')
        while np.linalg.norm(goal_ee[:3] - current_ee[:3]) > 1e-3 or np.sum(np.abs(qd)) > 1e-3:
            current_ee, qd = self.get_feedback('tool_pose', 'joint_velocities')
            time.sleep(0.01)
