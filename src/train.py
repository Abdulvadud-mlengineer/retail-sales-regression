import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# загрузка данных
df = pd.read_csv("data/retail_sales.csv")

# пример: target = Sales
X = df.drop("Sales", axis=1)
y = df["Sales"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
preds = model.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print("MAE:", mae)
print("MSE:", mse)

# save model
joblib.dump(model, "models/model.pkl")
