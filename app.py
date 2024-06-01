##Make sure these dependecies have been installed: pip install numpy pandas matplotlib scikit-learn tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the CSV file into a DataFrame
data = pd.read_csv('amazon.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Visualize the data
plt.figure(figsize=(14, 5))
plt.plot(data['Close'])
plt.title('Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Ensure the data is sorted by date
data = data.sort_values('Date')

# Select the 'Close' column for prediction
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define training and testing data sizes
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1, :] = test_predict

dates = pd.to_datetime(data['Date'])



# Number of Monte Carlo samples
num_mc_samples = 5  # You can adjust this

# Make predictions with Monte Carlo Dropout
def get_mc_predictions(model, X_test, num_samples):
    predictions = []
    for _ in range(num_samples):
        # Create a clone of the model to isolate the dropout layers
        cloned_model = clone_model(model)
        cloned_model.set_weights(model.get_weights())
        
        # Predict using the cloned model with dropout
        predictions.append(cloned_model.predict(X_test))  
    return np.array(predictions)


mc_test_predictions = get_mc_predictions(model, X_test, num_mc_samples)

# Calculate mean and standard deviation across predictions
mean_predictions = np.mean(mc_test_predictions, axis=0)
std_dev_predictions = np.std(mc_test_predictions, axis=0)

# Plot each prediction individually
for i in range(num_mc_samples):
    plt.figure(figsize=(14, 5))

    # Plot baseline (actual prices)
    plt.plot(dates, close_prices, label='Actual')
    plt.plot(dates[time_step : len(train_predict) + time_step], train_predict_plot[time_step : len(train_predict) + time_step], label='Train Prediction (80%)')
        
        
    
    # Plot the individual Monte Carlo prediction
    test_predict_plot = np.empty_like(close_prices)
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1, :] = mc_test_predictions[i]

    plt.plot(
        dates[len(train_predict) + (time_step * 2) + 1 : len(scaled_data) - 1],
        scaler.inverse_transform(test_predict_plot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1]),
        label=f'Test Prediction {i+1}',
        color='green'
    )


    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc='upper left')
    
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()

    plt.show()


# Plot mean prediction
plt.figure(figsize=(14, 5))
plt.plot(dates, close_prices, label='Actual')
plt.plot(
    dates[len(train_predict) + (time_step * 2) + 1 : len(scaled_data) - 1],
    scaler.inverse_transform(mean_predictions.reshape(-1, 1)),
    label='Mean Test Prediction',
    color='purple'
)
plt.title('Mean Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.show()

# Print mean prediction
print("Mean Prediction:")
print(scaler.inverse_transform(mean_predictions.reshape(-1, 1)))
print("\n")
