import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import datetime as dt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the data
df = pd.read_csv('data.csv')

# Filter data for Greece
df = df[df['Entity'] == 'Greece']

# Convert Date to ordinal
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.datetime.toordinal)

# Calculate Positivity Rate
df['Positivity Rate'] = df['Cases'] / df['Daily tests']

# Replace infinities or NaNs with a specific value or method of your choice
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Replace infinities or NaNs with a specific value or method of your choice
df['Positivity Rate'] = df['Positivity Rate'].replace([np.inf, -np.inf], np.nan)
df['Positivity Rate'].fillna(method='ffill', inplace=True)

# We would create a column that represents the target variable 3 days in the future.
# Note that this operation assumes that your data is ordered by date.
df['Positivity Rate 3 Days Later'] = df['Positivity Rate'].shift(-3)

# Remove rows with missing targets
df = df.dropna(subset=['Positivity Rate 3 Days Later'])

# Drop non-numerical columns if any exist
df = df.select_dtypes(include=[np.number])

df.to_csv('greece_data.csv', index=False)
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('Positivity Rate 3 Days Later', axis=1)
y = df['Positivity Rate 3 Days Later']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the SVM model
svm_regressor = svm.SVR()
svm_regressor.fit(X_train_scaled, y_train)

# Evaluate the SVM model
svm_predictions = svm_regressor.predict(X_test_scaled)
print('SVM MSE:', mean_squared_error(y_test, svm_predictions))


# Print SVM predictions alongside actual values
print("\nSVM Predictions vs Actual values:")
for predicted, actual in zip(svm_predictions, y_test):
    print(f"Predicted: {predicted}, Actual: {actual}")


# For RNN, we need to reshape input to be [samples, time steps, features]
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Create the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(4, input_shape=(1, len(X_train.columns))))
rnn_model.add(Dense(1))
rnn_model.compile(loss='mean_squared_error', optimizer='adam')
rnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=1, verbose=2)

# Evaluate the RNN model
rnn_predictions = rnn_model.predict(X_test_scaled)
print('RNN MSE:', mean_squared_error(y_test, rnn_predictions))


# Print RNN predictions alongside actual values
print("\nRNN Predictions vs Actual values:")
for predicted, actual in zip(rnn_predictions.flatten(), y_test):
    print(f"Predicted: {predicted}, Actual: {actual}")

