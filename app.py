from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model khi start server
model = joblib.load("models/iris_model.pkl")

@app.get("/")
def home():
    return {"message": "Iris model API is running"}

@app.post("/predict")

def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
