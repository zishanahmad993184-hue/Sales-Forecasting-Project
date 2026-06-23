import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("sales_data.csv")

X = data[['Month']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({'Month':[13,14,15,16,17,18]})

predictions = model.predict(future_months)

print("Future Sales Forecast")

for month, sale in zip(future_months['Month'], predictions):
    print(f"Month {month}: {sale:.2f}")

plt.scatter(X, y)

plt.plot(future_months, predictions)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting")

plt.savefig("forecast_graph.png")

plt.show()