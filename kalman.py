
import numpy as np


class Kalman:
    """Generic Kalman filter in Python."""

    def __init__(self, F, G, H, P, Q, R, X):
        """Initialize the state matrixes."""
        self.F, self.G, self.H = F, G, H
        self.P, self.Q, self.R = P, Q, R
        self.X = X
        self.K = None

    def predict(self, U):
        """Predict the next state from the current state X
        and a control vector U."""
        return self.F @ self.X + self.G @ U

    def update(self, X, Z):
        """Update the internal state and return an updated
        state estimate after estimating the current filter
        gain K. X is the predicted state, Z is the current
        measurement."""
        P = (self.F @ (self.P @ self.F.T)) + self.Q
        Y = Z - self.H @ X
        S = self.H @ P @ self.H.T + self.R
        self.K = P @ self.H.T @ np.linalg.inv(S)
        self.X = X + self.K @ Y
        T = self.K @ self.H
        I = np.eye(T.shape[0])
        self.P = (I - T) @ (P @ (I - T).T) + self.K @ (self.R @ self.K.T)
        return self.H @ self.X


def test0():
    sigma = 2.0
    F = np.matrix([[1.0]])
    G = np.matrix([[0.0]])
    H = np.matrix([[1.0]])
    kf = Kalman(F, G, H, np.zeros((1, 1)),
                np.diag([sigma ** 2]),
                np.diag([sigma ** 2]),
                np.array([0.0]))
    T = np.random.default_rng()
    Y = np.matrix([[T.normal(scale=sigma)] for _ in range(3000)])
    for Z in Y:
        X = kf.predict(Z)
        kf.update(X, Z)
        print(f'Z = {Z}, X = {X}, kf.X = {kf.X} kf.K = {kf.K}')


if __name__ == "__main__":
    test0()
