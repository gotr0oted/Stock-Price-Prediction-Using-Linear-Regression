import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download("TSLA", start="2020-01-01", end="2024-01-01", auto_adjust=True)

print(data.head())

plt.plot(data['Close'])
plt.title("Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

data['Prediction'] = data['Close'].shift(-30)

X = np.array(data[['Close']])[:-30]
y = np.array(data['Prediction'])[:-30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

X_future = np.array(data[['Close']])[-30:]
predictions = model.predict(X_future)

print("Predicted prices:", predictions)
print("Model Accuracy:", model.score(X_test, y_test))

sorted_indices = np.argsort(X_test.flatten())

plt.figure(figsize=(10,5))
plt.plot(y_test[sorted_indices], label="Actual")
plt.plot(model.predict(X_test)[sorted_indices], label="Predicted")

plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show()
