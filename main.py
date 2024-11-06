import numpy as np
import matplotlib.pyplot as plt
from data_stream_generators import simulate_data_stream, introduce_anomalies
from kalman import KalmanFilter

# Initialize Kalman filter parameters
A = np.array([[1]])     # State transition matrix
B = np.array([[0]])     # Control matrix
H = np.array([[1]])     # Observation matrix
Q = np.array([[1e-3]])  # Process noise covariance, Increase Q to make the filter adapt more quickly to abrupt changes in the system or data
R = np.array([[1e-2]])  # Measurement noise covariance, A larger R tells the filter that the measurements are noisy and should be trusted less
P = np.array([[1]])     # Initial error covariance
x0 = np.array([[0]])    # Initial state estimate

kf = KalmanFilter(A, B, H, Q, R, P, x0)

WINDOW_SIZE = 100   # Number of data points to display in the plot
PAUSE_TIME = 0.1    # Pause time in seconds for the plot TO SIMULATE REAL TIME PROCESSING
BASE = 10           # Base level of the data stream
Y_MAX = BASE + 10   # Initial y-axis limit for the plot
THRESHOLD = 3       # Anomaly detection threshold, considered anomaly data point is greater than prediction by threshold


data = np.array([BASE])      # Array to store true data points
predictions = np.array([BASE])  #Initial prediction of kalman filter is set to 0
anomalies = np.array([])     # Array to store detected anomalies


data_stream = simulate_data_stream(length=150, base_level=BASE, season_length=12, noise_level=0.25, trend_slope=0.1)
data_stream = introduce_anomalies(data_stream, num_anomalies=5, anomaly_value=5)
data_stream = introduce_anomalies(data_stream, num_anomalies=5, anomaly_value=-5)

kf = KalmanFilter(A, B, H, Q, R, P, x0)

# Initialize the plot
plt.ion()                   # Interactive mode on
fig, ax = plt.subplots()
line, = ax.plot(data, label='True Data', color='blue')
prediction_line, = ax.plot(predictions, label='Predictions', color='orange')
anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')
ax.set_xlim(0, WINDOW_SIZE)
ax.set_ylim(-10, 30)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("Real-time Data Stream and Prediction with Anomaly Detection")
ax.legend()

# Real-time data processing and plotting
for i, data_point in enumerate(data_stream):
    data = np.append(data, data_point)
    
    # Prediction step
    predicted_state = kf.predict()
    
    # Update step
    updated_state = kf.update(data_point)
    
    residual = np.abs(data_point - predicted_state[0])
    if residual > THRESHOLD:
        anomalies = np.append(anomalies, data_point)
    
    predictions = np.append(predictions, updated_state[0])

    # Keep only the last WINDOW_SIZE data points
    if len(data) > WINDOW_SIZE:
        data = data[-WINDOW_SIZE:]
        predictions = predictions[-WINDOW_SIZE:]

    # Adjust y-axis limits dynamically
    Y_MAX = max(Y_MAX, np.max(data)+1, np.max(predictions)+1)
    ax.set_ylim(BASE-10, Y_MAX)

    # Update plot data
    line.set_data(range(len(data)), data)
    prediction_line.set_data(range(len(predictions)), predictions)
    anomaly_indices = np.where(np.abs(data - predictions) > THRESHOLD)[0]
    anomaly_points.set_data(anomaly_indices, data[anomaly_indices])


    plt.draw()
    plt.pause(PAUSE_TIME)

plt.ioff()  # Turn off interactive mode
plt.show()