import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        """
        Initialize the Kalman filter parameters.
        
        A: State transition matrix
        B: Control matrix
        H: Observation matrix
        Q: Process noise covariance matrix
        R: Observation noise covariance matrix
        P: Covariance matrix
        x0: Initial state estimate
        """
        self.A = A  # State transition model
        self.B = B  # Control input model
        self.H = H  # Observation model
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Error covariance
        self.x = x0  # Initial state estimate
    
    def predict(self, u=0):
        """
        Predict the next state and covariance.
        u: Control input (optional)
        """
        # Predict the state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        # Predict the error covariance
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return self.x
    
    def update(self, z):
        """
        Update the state and covariance using the observation.
        z: Observation
        """
        # Compute the Kalman gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update the state estimate
        y = z - np.dot(self.H, self.x)  # Measurement residual
        self.x = self.x + np.dot(K, y)
        
        # Update the error covariance
        I = np.eye(self.A.shape[0])
        self.P = (I - np.dot(K, self.H)) @ self.P
        
        return self.x
