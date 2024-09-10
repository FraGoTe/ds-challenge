from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from challenge.model import DelayModel

import pandas as pd

app = FastAPI()

@app.get("/", status_code=200)
async def get_health() -> dict:
    return {
        "status": "hello world 🚀"
    }

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictionInput(BaseModel):
    flights: List[Flight]

@app.post("/predict", status_code=200)
async def post_predict(input_data: PredictionInput) -> Dict:

    delay_model = DelayModel()
    delay_model.load_model('./../data/xgboost_model.pkl')
  
    flight_dicts = []
    predictions = []
    for flight in input_data.flights:

        if flight.MES < 0 or flight.MES > 12:
            raise HTTPException(
                status_code=400, detail=f"Invalid MES sent")


        flight_data =  pd.concat([
                pd.get_dummies(flight.OPERA, prefix = 'OPERA'),
                pd.get_dummies(flight.TIPOVUELO, prefix = 'TIPOVUELO'), 
                pd.get_dummies(flight.MES, prefix = 'MES')], 
                axis = 1
                )
        flight_data = flight_data.reindex(columns = delay_model.get_top_10_features(), fill_value = False)
        predictions.extend(delay_model.predict(flight_data.head(1)))
   
    return {
        'predict': predictions
    }