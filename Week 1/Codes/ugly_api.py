import uvicorn

import numpy as np
import pandas as pd

from fastapi import FastAPI
from catboost import CatBoostClassifier


#Load model function
def load_model():
    
    model= CatBoostClassifier()
    model.load_model('artifacts/scoring_model.cbm')
    
    return model


MODEL= load_model()


def get_score(request, model=MODEL):
    
    dataframe= pd.DataFrame.from_dict(request,orient='index').T
    score    = model.predict_proba(dataframe)[0][1]
    
    return score

app= FastAPI(title="patient-api", version="0.1")
             
@app.post('/check_patient/')
def check_patient(patient:dict):
    
    return{'result': get_score(patient)}

if __name__ == '__main__':
    uvicorn.run("ugly_api:app", host='127.0.0.1', port=5000)