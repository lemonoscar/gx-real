from operator import le
import time
from typing import List

import numpy as np
import collections


"""Moving window filter to smooth out sensor readings. From https://github.com/erwincoumans/motion_imitation"""

class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.
    We implement the Neumaier's algorithm to calculate the moving window average,
    which is numerically stable.
    """

    def __init__(self, window_size: int, data_dim: int):
        """Initializes the class.

        Args:
          window_size: The moving window size.
        """
        assert window_size > 0
        self._window_size: int = window_size
        self._data_dim = data_dim
        # self._value_deque = collections.deque(maxlen=window_size)
        # Use numpy array to simulate deque so that it can be compiled by numba
        self._value_deque = np.zeros((self._data_dim, window_size), dtype=np.float64)
        # The moving window sum.
        self._sum = np.zeros((self._data_dim,), dtype=np.float64)
        # The correction term to compensate numerical precision loss during
        # calculation.
        self._correction = np.zeros((self._data_dim,), dtype=np.float64)

    def _neumaier_sum(self, value):
        """Update the moving window sum using Neumaier's algorithm.

        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

        Args:
          value: The new value to be added to the window.
        """
        assert value.shape == (self._data_dim,)
        new_sum = self._sum + value
        for k in range(self._data_dim):
            if abs(self._sum[k]) >= abs(value[k]):
                # If self._sum is bigger, low-order digits of value are lost.
                self._correction[k] += (self._sum[k] - new_sum[k]) + value[k]
            else:
                # low-order digits of sum are lost
                self._correction[k] += (value[k] - new_sum[k]) + self._sum[k]

        self._sum = new_sum

    def calculate_average(
        self, new_value
    ):
        """Computes the moving window average in O(1) time.

        Args:
          new_value: The new value to enter the moving window.

        Returns:
          The average of the values in the window.

        """
        assert new_value.shape == (self._data_dim,)

        self._neumaier_sum(-self._value_deque[:, 0])
        self._neumaier_sum(new_value)

        # self._value_deque.append(new_value)
        for i in range(self._data_dim):
            self._value_deque[i, :] = np.roll(self._value_deque[i, :], -1)
        self._value_deque[:, -1] = new_value

        return (self._sum + self._correction) / self._window_size


def analytical_leg_jacobian(
    leg_angles,
    leg_id: int,
    hip_length: float,
    thigh_length: float,
    calf_length: float,
):
    """
    Computes the analytical Jacobian.
    Args:
        leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
        l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    assert len(leg_angles) == 3
    assert leg_id in [0, 1, 2, 3]

    hip_angle, thigh_angle, calf_angle = leg_angles[0], leg_angles[1], leg_angles[2]

    # Compute the effective length of the leg
    leg_length_eff = np.sqrt(
        thigh_length**2
        + calf_length**2
        + 2 * thigh_length * calf_length * np.cos(calf_angle)
    )
    leg_angle_eff = thigh_angle + calf_angle / 2

    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -leg_length_eff * np.cos(leg_angle_eff)
    J[0, 2] = (
        calf_length
        * thigh_length
        * np.sin(calf_angle)
        * np.sin(leg_angle_eff)
        / leg_length_eff
        - leg_length_eff * np.cos(leg_angle_eff) / 2
    )
    J[1, 0] = -hip_length * np.sin(hip_angle) + leg_length_eff * np.cos(
        hip_angle
    ) * np.cos(leg_angle_eff)
    J[1, 1] = -leg_length_eff * np.sin(hip_angle) * np.sin(leg_angle_eff)
    J[1, 2] = (
        -calf_length
        * thigh_length
        * np.sin(hip_angle)
        * np.sin(calf_angle)
        * np.cos(leg_angle_eff)
        / leg_length_eff
        - leg_length_eff * np.sin(hip_angle) * np.sin(leg_angle_eff) / 2
    )
    J[2, 0] = hip_length * np.cos(hip_angle) + leg_length_eff * np.sin(
        hip_angle
    ) * np.cos(leg_angle_eff)
    J[2, 1] = leg_length_eff * np.sin(leg_angle_eff) * np.cos(hip_angle)
    J[2, 2] = (
        calf_length
        * thigh_length
        * np.sin(calf_angle)
        * np.cos(hip_angle)
        * np.cos(leg_angle_eff)
        / leg_length_eff
        + leg_length_eff * np.sin(leg_angle_eff) * np.cos(hip_angle) / 2
    )
    return J


def inv_with_jit(M):
    return np.linalg.inv(M)


class VelocityEstimator:
    """Estimates base velocity of A1 robot.

    The velocity estimator consists of 2 parts:
    1) A state estimator for CoM velocity.

    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.

    2) A moving average filter to smooth out velocity readings
    """

    def __init__(
        self,
        hip_length: float,
        thigh_length: float,
        calf_length: float,
        accelerometer_variance=0.1,
        sensor_variance=0.1,
        initial_variance=0.1,
        moving_window_filter_size=120,
        default_control_dt=0.002,
    ):
        """Compatibility estimator for Jetson deployment.

        The original UMI estimator depends on filterpy/scipy, which is fragile on
        the robot's system Python stack. For deployment bootstrap we keep the same
        interface but output a conservative zero body linear velocity estimate.
        """

        self._initial_variance = initial_variance
        self._window_size = moving_window_filter_size
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )
        self._estimated_velocity = np.zeros(3)
        self._last_timestamp_s = 0.0
        self._default_control_dt = default_control_dt
        self.hip_length = hip_length
        self.thigh_length = thigh_length
        self.calf_length = calf_length

    def reset(self):
        self._estimated_velocity = np.zeros(3)
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )
        self._last_timestamp_s = 0.0

    def _compute_delta_time(self, new_timestamp_s: float):
        if self._last_timestamp_s == 0.0:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self._default_control_dt
        else:
            delta_time_s = new_timestamp_s - self._last_timestamp_s
        self._last_timestamp_s = new_timestamp_s
        return delta_time_s

    def update(
        self,
        new_timestamp_s: float,
        acceleration,
        foot_contact,
        quaternion,
        joint_velocity,
        joint_position,
    ):
        """Keep interface-compatible updates with conservative zero velocity."""
        assert acceleration.shape == (3,)
        assert foot_contact.shape == (4,)
        assert quaternion.shape == (4,)
        assert joint_velocity.shape == (12,)

        self._compute_delta_time(new_timestamp_s)
        self._estimated_velocity = self.moving_window_filter.calculate_average(
            np.zeros(3, dtype=np.float64)
        )

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()
