"""Stateful 483-D actor input. No ROS, Torch, privileged mass or hardware I/O.

History contains four PREVIOUS unscaled frames, oldest first. Advance only at
50 Hz inference, never on low-state callbacks. Quaternions are wxyz; odometry
must describe the base origin (not LiDAR origin) in one continuous world frame.
"""
import numpy as np

MDP = "robot_lab.tasks.manager_based.locomotion.velocity.mdp."
MEASUREMENT = MDP + "unified_asymmetric:actor_measurement"
FEATURES = MDP + "unified_asymmetric:actor_features"
PROPRIO_PROFILE = "proprioceptive_flat_v1"
PROPRIO_FEATURES = MDP + "proprioceptive_flat:proprio_features"
PROPRIO_VELOCITY = MDP + "proprioceptive_flat:estimated_velocity"
PROPRIO_HEIGHT = MDP + "proprioceptive_flat:zero_height"
MEASUREMENTS = {"base_lin_vel": ("lin", .1, 2.), "base_ang_vel": ("ang", .02, .25),
                "projected_gravity": ("gravity", .02, 1.), "joint_pos": ("q", .01, 1.),
                "joint_vel": ("dq", 1.5, .05)}


def vector(value, size):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"expected finite vector of size {size}")
    return value


def filter_command(command):
    c = vector(command, 3).copy()
    dead = np.array([.1, .05, .05])
    c[np.abs(c) <= dead] = 0
    limits = np.array([.5 if c[0] < 0 else .75, .2, .3])
    c /= max(1., np.linalg.norm(c / limits))
    c[np.abs(c) <= dead] = 0
    return c


def rotation(quaternion):
    q = vector(quaternion, 4)
    norm = np.linalg.norm(q)
    if not .99 <= norm <= 1.01:
        raise ValueError("invalid base orientation quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class UnifiedPolicyInput:
    def __init__(self, home, use_xy_anchor=True):
        self.use_xy_anchor = bool(use_xy_anchor)
        self.home = vector(home, 6).copy()
        self.reset()

    def reset(self):
        self.frames = None
        self.phase = 0.
        self.released = False
        self.was_standing = False
        self.anchor = np.zeros(3)
        self.yaw_anchor = 0.
        self.last_step = None
        self.output = None

    def build(self, prefix, requested, arm_q, arm_dq, arm_target, position, quaternion, step):
        prefix = vector(prefix, 260).copy()
        position = vector(position, 3) if self.use_xy_anchor else None
        rot = rotation(quaternion)
        q = vector(arm_q, 6)
        dq = vector(arm_dq, 6)
        target = vector(arm_target, 6)
        command = filter_command(requested)
        if step == self.last_step:
            return self.output.copy()
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError("policy step discontinuity: reset history before restarting")
        error = np.max(np.abs(q-self.home))
        target_ready = np.max(np.abs(target-self.home)) < .03
        ready = error < .1 and target_ready and np.max(np.abs(dq)) < .15
        self.released = (self.released or ready) and np.any(command != 0) and target_ready and error < .20
        if not self.released:
            command[:] = 0
        standing = not np.any(command)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        if standing and not self.was_standing:
            if self.use_xy_anchor:
                self.anchor = position.copy()
            self.yaw_anchor = yaw
        demand = np.clip(np.linalg.norm(command[:2])/.75 + abs(command[2])/.6, 0, 1)
        self.phase = 0. if standing else (self.phase + .02*(.7+1.3*demand)) % 1
        self.was_standing = standing
        delta = rot.T @ (self.anchor-position) if self.use_xy_anchor else np.zeros(3)
        dyaw = yaw-self.yaw_anchor
        features = ([0, 0, 1, delta[0], delta[1], np.sin(dyaw), np.cos(dyaw)-1] if standing else
                    [np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase), 0, 0, 0, 0, 0])
        prefix[9:12] = command
        prefix[253:259] = target
        prefix[259] = 0.  # Training constant; physical gripper target is not an observation.
        # Outer observation scales do not apply inside the training history function.
        frame = np.concatenate([prefix[3:6]/.25, prefix[6:9], prefix[12:30],
                                prefix[30:48]/.05, prefix[48:60]])
        if self.frames is None:
            self.frames = np.tile(frame, (4, 1))
        self.output = np.concatenate([prefix, features, self.frames.ravel()]).astype(np.float32)
        self.frames = np.concatenate([self.frames[1:], frame[None]], axis=0)
        self.last_step = step
        return self.output.copy()
