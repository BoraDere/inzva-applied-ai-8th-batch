import time
import uvicorn

import numpy as np
import pandas as pd

from fastapi import FastAPI
from catboost import CatBoostClassifier
from data_models import Patient, Response

#Load model function
def load_model():
    
    model= CatBoostClassifier()
    model.load_model('artifacts/model_name.cbm')
    
    return model


MODEL= load_model()


def get_score(request, model=MODEL):
    start_time = time.time()
    dataframe  = pd.DataFrame.from_dict(vars(request),orient='index').T
    score      = model.predict_proba(dataframe)[0][1]
    return {'Score':score,
            'Class':np.round(score),
            'runtime': np.round((time.time() - start_time)*1000,2)}

app= FastAPI(title="patient-api", version="0.1")
             
@app.post('/check_patient/',response_model=Response)
def check_patient(patient:Patient)->Response:
    
    return Response(**get_score(patient))

if __name__ == '__main__':
    uvicorn.run("fancy_api:app", host='0.0.0.0', port=5000)