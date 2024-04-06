from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Soil_color: int
    Nitrogen: int
    Phosphorus: int
    Potassium: int
    pH: float
    Temperature: int
    Crop: int

fertilizer_model = joblib.load('crop_ferti.joblib')


@app.post('/predict')
async def predict_fertilizer(data: ModelInput):
    # Extract individual features
    Soil_color = data.Soil_color
    Nitrogen = data.Nitrogen
    Phosphorus = data.Phosphorus
    Potassium = data.Potassium
    pH = data.pH
    Temperature = data.Temperature
    Crop = data.Crop

    # Make prediction
    prediction = fertilizer_model.predict(
        [[Soil_color, Nitrogen, Phosphorus, Potassium, pH, Temperature, Crop]])

    # Convert the prediction to a string
    prediction_str = str(prediction[0])

    return {'Fertilizer_prediction': prediction_str}  # Convert prediction to list for JSON serialization

