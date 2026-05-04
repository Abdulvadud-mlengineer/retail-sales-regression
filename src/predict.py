import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

# пример входных данных (позже заменишь под свой датасет)
sample = pd.DataFrame([{
    "feature1": 10,
    "feature2": 5
}])

prediction = model.predict(sample)

print("Predicted sales:", prediction[0])
