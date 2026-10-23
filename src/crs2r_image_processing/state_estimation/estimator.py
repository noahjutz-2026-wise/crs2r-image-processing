import numpy as np
from typing import Tuple


class FiniteDiff:
    """
    State estimation using finite difference
    """

    def __init__(self, fps: float, FiniteDiff_mean_steps: int) -> None:
        self.fps = fps
        self.T_s = 1.0 / self.fps
        self.mean_steps = FiniteDiff_mean_steps
        self.prev_measurements = np.zeros((self.mean_steps, 2))

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        x_0 = np.array([0.014, 0.104, 0, 0])
        P_0 = np.eye(4)

        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prev_mean = self.prev_measurements.mean(axis=0)
        self.prev_measurements = np.concatenate(
            (self.prev_measurements[-self.mean_steps + 1 :], measurement.reshape(1, 2)),
            axis=0,
        )
        curr_mean = self.prev_measurements.mean(axis=0)
        finite_diff = (curr_mean - prev_mean) / self.T_s

        x = np.hstack((measurement, finite_diff))

        self.prev_measurement = measurement

        return x, np.eye(4)


class KF:
    """
    Kalman Filter class
    """

    def __init__(self, fps: float):
        self.g = 9.81
        self.fps = fps
        self.T_s = 1 / self.fps

        # DT linear model dynamics
        self.A = np.array(
            [[1, 0, self.T_s, 0], [0, 1, 0, self.T_s], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.B = np.array(
            [
                [0, 5 / 14 * self.g * self.T_s ** 2],
                [-5 / 14 * self.g * self.T_s ** 2, 0],
                [0, 5 / 7 * self.g * self.T_s],
                [-5 / 7 * self.g * self.T_s, 0],
            ]
        )

        # process noise v ~ N(0, Q),  Q: [4x4]
        self.sigma_angle = (
            np.pi / 180 * 10
        )  # noise on the angles measurements (here treated as inputs) (alpha, beta)
        self.Q = (
            self.B @ np.diag([self.sigma_angle ** 2, self.sigma_angle ** 2]) @ self.B.T
        )

        # DT linear measurement model
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # measurement noise w ~ N(0, R), R: [2x2]
        self.sigma_m = 1e-4
        self.R = np.diag([self.sigma_m ** 2, self.sigma_m ** 2])

        self.xm, self.Pm = self.initialize()

    def initialize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the initial state estimate. The order of states is
                given by x = [xb, yb, xb_dot, yb_dot].
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the initial state estimate. The order of
                states is given by x = xb, yb, xb_dot, yb_dot].
        """
        self.sigma_p0 = 1
        self.sigma_v0 = 1
        x_0 = np.array([0.014, 0.104, 0, 0])
        P_0 = np.diag(
            [
                self.sigma_p0 ** 2,
                self.sigma_p0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_v0 ** 2,
            ]
        )
        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            inputs : np.ndarray, dim: (2,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [alpha, beta].
            measurement : np.ndarray, dim: (2,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [xb, yb].

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [xb, yb, xb_dot, yb_dot].
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the posterior estimate Pm(k). The order of
                states is given by x = xb, yb, xb_dot, yb_dot].
        """
        # prior update
        xp, Pp = self.prior_update(self.xm, self.Pm, inputs)

        # measurement update
        self.xm, self.Pm = self.measurement_update(xp, Pp, measurement)

        return self.xm, self.Pm

    def prior_update(self, xm_prev: np.ndarray, Pm_prev: np.ndarray, u: np.ndarray):

        xp = self.A @ xm_prev + self.B @ u
        Pp = self.A @ Pm_prev @ self.A.T + self.Q
        return xp, Pp

    def measurement_update(
        self,
        xp: np.ndarray,
        Pp: np.ndarray,
        measurement: np.ndarray,
    ):

        if np.any(np.isnan(measurement)):
            return xp, Pp

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)
        # mean update
        z_pred = self.H @ xp
        xm = xp + K @ (measurement - z_pred)
        # variance update
        Pm = (np.eye(xp.shape[0]) - K @ self.H) @ Pp @ (
            np.eye(xp.shape[0]) - K @ self.H
        ).T + K @ self.R @ K.T
        return xm, Pm


class KFBias:
    """
    Kalman Filter class
    """

    def __init__(self, fps: float):
        self.g = 9.81
        self.fps = fps
        self.T_s = 1 / self.fps

        T_s = self.T_s
        g = self.g
        # DT linear model dynamics
        self.A = np.array(
            [
                [1, 0, T_s, 0, 0, (5 * T_s ** 2 * g) / 14],
                [0, 1, 0, T_s, -(5 * T_s ** 2 * g) / 14, 0],
                [0, 0, 1, 0, 0, (5 * T_s * g) / 7],
                [0, 0, 0, 1, -(5 * T_s * g) / 7, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        self.B = np.array(
            [
                [0, 5 / 14 * g * T_s ** 2],
                [-5 / 14 * g * T_s ** 2, 0],
                [0, 5 / 7 * g * T_s],
                [-5 / 7 * g * T_s, 0],
                [0, 0],
                [0, 0],
            ]
        )

        # process noise v ~ N(0, Q),  Q: [4x4]
        self.sigma_angle = (
            np.pi / 180 * 1
        )  # noise on the angles measurements (here treated as inputs) (alpha, beta)
        self.Q = (
            self.B @ np.diag([self.sigma_angle ** 2, self.sigma_angle ** 2]) @ self.B.T
        )

        # DT linear measurement model
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        # measurement noise w ~ N(0, R), R: [2x2]
        self.sigma_m = 1e-4
        self.R = np.diag([self.sigma_m ** 2, self.sigma_m ** 2])

        self.xm, self.Pm = self.initialize()

    def initialize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (6,)
                The mean of the initial state estimate. The order of states is
                given by x = [xb, yb, xb_dot, yb_dot, alpha_bias, beta_bias].
            Pm : np.ndarray, dim: (6, 6)
                The covariance of the initial state estimate.
        """
        self.sigma_p0 = 1
        self.sigma_v0 = 1
        self.sigma_bias0 = 0.1
        x_0 = np.array([0.014, 0.104, 0, 0, 0, 0])
        P_0 = np.diag(
            [
                self.sigma_p0 ** 2,
                self.sigma_p0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_bias0 ** 2,
                self.sigma_bias0 ** 2,
            ]
        )
        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            inputs : np.ndarray, dim: (2,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [alpha, beta].
            measurement : np.ndarray, dim: (2,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [xb, yb].

        Returns:
            xm : np.ndarray, dim: (6,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [xb, yb, xb_dot, yb_dot, alpha_bias, beta_bias].
            Pm : np.ndarray, dim: (6, 6)
                The covariance of the posterior estimate Pm(k).
        """
        # prior update
        xp, Pp = self.prior_update(self.xm, self.Pm, inputs)

        # measurement update
        self.xm, self.Pm = self.measurement_update(xp, Pp, measurement)

        return self.xm, self.Pm

    def prior_update(self, xm_prev: np.ndarray, Pm_prev: np.ndarray, u: np.ndarray):

        xp = self.A @ xm_prev + self.B @ u
        Pp = self.A @ Pm_prev @ self.A.T + self.Q
        return xp, Pp

    def measurement_update(
        self,
        xp: np.ndarray,
        Pp: np.ndarray,
        measurement: np.ndarray,
    ):

        if np.any(np.isnan(measurement)):
            return xp, Pp

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)
        # mean update
        z_pred = self.H @ xp
        xm = xp + K @ (measurement - z_pred)
        # variance update
        Pm = (np.eye(xp.shape[0]) - K @ self.H) @ Pp @ (
            np.eye(xp.shape[0]) - K @ self.H
        ).T + K @ self.R @ K.T
        return xm, Pm
