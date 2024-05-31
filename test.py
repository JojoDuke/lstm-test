
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data
# Replace with your actual data
data = np.random.rand(100, 1)
labels = np.sin(data).flatten()

# Split data into training and testing sets
train_data, test_data = data[:80], data[80:]
train_labels, test_labels = labels[:80], labels[80:]

# Define LSTM model with dropout
model = keras.Sequential([
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(None, 1)),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1)
])

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(train_data, train_labels, epochs=50, batch_size=16, verbose=0)

# Predictions
predictions = model.predict(test_data)

# Calculate uncertainty using dropout during inference
mc_samples = 100
predictions_mc = np.stack([model.predict(test_data) for _ in range(mc_samples)], axis=0)
mean_prediction = np.mean(predictions_mc, axis=0)
stddev_prediction = np.std(predictions_mc, axis=0)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(test_data, test_labels, label='True values', color='blue')
plt.plot(test_data, mean_prediction, label='Mean Prediction', color='red')
plt.fill_between(test_data.flatten(), 
                 mean_prediction.flatten() - 2 * stddev_prediction.flatten(),
                 mean_prediction.flatten() + 2 * stddev_prediction.flatten(),
                 color='orange', alpha=0.3, label='Uncertainty Bounds')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('LSTM Prediction with Dropout Uncertainty')
plt.legend()
plt.show()

