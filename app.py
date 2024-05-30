import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##from sklearn.preprocessing import MinMaxScaler

# Load the CSV file into a DataFrame
data = pd.read_csv('./access_shares.csv')

# Check the first few rows of the DataFrame to ensure it's loaded correctly
print(data.head())

# Visualize the data
plt.figure(figsize=(14, 5))
plt.plot(data['Close'])
plt.title('Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

