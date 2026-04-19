import numpy as np
 
"""Moving window filter to smooth out sensor readings. From motion_imitation."""

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
        self._count = 0

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

        self._count = min(self._count + 1, self._window_size)
        return (self._sum + self._correction) / max(self._count, 1)


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


class KalmanFilter3D:
    def __init__(
        self,
        accelerometer_variance: float,
        sensor_variance: float,
        initial_variance: float,
    ):
        self.initial_variance = float(initial_variance)
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * self.initial_variance
        self.Q = np.eye(3, dtype=np.float64) * float(accelerometer_variance)
        self.R = np.eye(3, dtype=np.float64) * float(sensor_variance)

    def reset(self):
        self.x.fill(0.0)
        self.P[:] = np.eye(3, dtype=np.float64) * self.initial_variance

    def predict(self, dt: float, acceleration: np.ndarray):
        self.x = self.x + acceleration * float(dt)
        self.P = self.P + self.Q

    def update(self, measurement: np.ndarray):
        innovation = measurement - self.x
        S = self.P + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ S_inv
        self.x = self.x + K @ innovation
        identity = np.eye(3, dtype=np.float64)
        i_minus_k = identity - K
        self.P = i_minus_k @ self.P @ i_minus_k.T + K @ self.R @ K.T


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q.astype(np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 1e-6:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(q)
    return rotation.T @ v


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
        """Numpy-only velocity estimator compatible with umi-on-legs behavior."""
        self.filter = KalmanFilter3D(
            accelerometer_variance=accelerometer_variance,
            sensor_variance=sensor_variance,
            initial_variance=initial_variance,
        )
        self._initial_variance = initial_variance
        self._window_size = moving_window_filter_size
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )
        self._estimated_velocity = np.zeros(3, dtype=np.float64)
        self._last_timestamp_s = 0.0
        self._default_control_dt = default_control_dt
        self.hip_length = hip_length
        self.thigh_length = thigh_length
        self.calf_length = calf_length
        self._accelerometer_bias = np.zeros(3, dtype=np.float64)
        self._velocity_bias = np.zeros(3, dtype=np.float64)

    def reset(self):
        self.filter.reset()
        self._estimated_velocity = np.zeros(3, dtype=np.float64)
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
        """Estimate body-frame linear velocity from IMU and contact leg kinematics."""
        assert acceleration.shape == (3,)
        assert foot_contact.shape == (4,)
        assert quaternion.shape == (4,)
        assert joint_velocity.shape == (12,)
        assert joint_position.shape == (12,)

        delta_time_s = self._compute_delta_time(new_timestamp_s)
        rot_mat = quaternion_to_rotation_matrix(quaternion)
        calibrated_acc = acceleration.astype(np.float64) - self._accelerometer_bias
        acc_world = rot_mat @ calibrated_acc + np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.filter.predict(delta_time_s, acc_world)

        observed_velocities = []
        for leg_id in range(4):
            if foot_contact[leg_id]:
                start = leg_id * 3
                leg_angles = joint_position[start : start + 3]
                jacobian = analytical_leg_jacobian(
                    leg_angles=leg_angles,
                    leg_id=leg_id,
                    hip_length=self.hip_length,
                    thigh_length=self.thigh_length,
                    calf_length=self.calf_length,
                )
                joint_velocities = joint_velocity[start : start + 3]
                leg_velocity_in_base = jacobian.dot(joint_velocities)
                base_velocity_in_base = -leg_velocity_in_base[:3]
                observed_velocities.append(rot_mat.dot(base_velocity_in_base))

        if observed_velocities:
            observed_velocities = np.mean(np.asarray(observed_velocities, dtype=np.float64), axis=0)
            self.filter.update(observed_velocities)

        smoothed_velocity_world = self.moving_window_filter.calculate_average(self.filter.x.copy())
        self._estimated_velocity = rotate_inverse(quaternion, smoothed_velocity_world) - self._velocity_bias

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()
