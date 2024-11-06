import numpy as np

def simulate_data_stream(length=100, base_level=10, season_length=12, noise_level=1, trend_slope=0.1):
    """
    Simulate a data stream with a trend, seasonal component, and noise.

    Parameters:
    - length (int): The number of data points to generate. Default is 100.
    - base_level (float): The base level of the data stream. Default is 10.
    - season_length (int): The length of the seasonal cycle. Default is 12.
    - noise_level (float): The standard deviation of the Gaussian noise. Default is 1.
    - trend_slope (float): The slope of the trend component. Default is 0.1.

    Returns:
    - np.ndarray: An array containing the simulated data stream.
    """
    data_stream = []
    for t in range(length):
        anomaly = 0
        seasonal_component = 1 * np.sin(2 * np.pi * t / season_length)
        noise = np.random.normal(0, noise_level)
        trend = trend_slope * t
        data_point = base_level + seasonal_component + noise + trend + anomaly
        data_stream.append(data_point)
    return np.array(data_stream)

def introduce_anomalies(data_stream, num_anomalies=5, anomaly_value=20):
    """
    Introduce anomalies to the data stream.
    
    Parameters:
    - data_stream: numpy array, the original data stream
    - num_anomalies: int, the number of anomalies to introduce
    - anomaly_value: float, the value to add to create an anomaly
    
    Returns:
    - numpy array, the data stream with anomalies
    """
    data_with_anomalies = data_stream.copy()
    anomaly_indices = np.random.choice(len(data_stream), num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        data_with_anomalies[idx] += anomaly_value
    
    anomaly_indices.sort()
    print(f"Anomalies introduced at indices: {anomaly_indices}")
    return data_with_anomalies