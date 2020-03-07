## UR5 Realtime client
Lightweight python library for controlling UR5 robot from Universal Robot family. Robot kinematics are calculated based on the following paper: [tech report](https://smartech.gatech.edu/handle/1853/50782)
### Requirements
* UR Control Box (simulated or real)
* Tested on UR Robot software 3.5.1 
### Try it out!
```python
>>> from realtime_client import RTClient
>>> import numpy as np

# connect to address 127.0.0.1 and port 30003
>>> rtc = RTClient('127.0.0.1', 30003)

# move end effector to joint goal [q1, q2, q3, q4, q5, q6]
>>> joint_values = [d * (np.pi / 180) for d in [-90, -90, -90, -90, 89, 5]]
>>> rtc.move_j(joint_values)

# move end effector to pose goal [x,y,z,rx,ry,rz] (3D translation and 3D rotation)
>>> pose = np.array([-0.46481069, -0.18235116,  0.13827986, -1.58136603, -2.69628063, -0.01169701])
>>> rtc.move_l(pose)

# velocity-based controller, move to pose goal [x,y,z,rx,ry,rz]
>>> pose = np.array([0.470, -0.491, 0.430, 0.13, 3.15, -0.00])
>>> rtc.move_v(pose)


# kinematic test
>>> from kinematics import KinematicsUR5
>>> from math_tools import pose2tf
>>> kin = KinematicsUR5()
>>> target_ee = np.array([-1.10586325e-01, -4.86899999e-01,  4.31871547e-01, -1.36273738e-01, -3.12118227e+00,  1.18929713e-03])
>>> target_pose = pose2tf(target_ee)
>>> solutions = kin.inv_kin(target_pose)
>>> closest_solution = kin.get_closest_solution(solutions, rtc.get_feedback('joint_values'))
>>> rtc.move_j(closest_solution)
>>> assert np.allclose(rtc.get_feedback('tool_pose'), target_ee)

>>> rtc.close_connection()


```