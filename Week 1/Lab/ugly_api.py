import uvicorn
import pandas as pd
from fastapi import FastAPI
from catboost import CatBoostClassifier

from pydantic import BaseModel
from enum import Enum

from datetime import timedelta
import datetime


class ResponseModel(BaseModel):
    score: float
    class_: str
    runtime: timedelta


class Age(int, Enum):
    age_5 = 5
    age_15 = 15
    age_25 = 25
    age_35 = 35
    age_45 = 45
    age_55 = 55
    age_65 = 65
    age_75 = 75
    age_85 = 85
    age_95 = 95


class Gender(str, Enum):
    female = "Female"
    male = "Male"
    unknown_invalid = "Unknown/Invalid"


class Race(str, Enum):
    caucasian = "Caucasian"
    african_american = "AfricanAmerican"
    other = "Other"
    asian = "Asian"
    hispanic = "Hispanic"


class Patient(BaseModel):
    age: Age
    gender: Gender
    race: Race = Race.other


def load_model():
    model= CatBoostClassifier()
    model.load_model('artifacts/scoring_model.cbm')
    
    return model


MODEL = load_model()


def get_score(request, model=MODEL):
    dataframe = pd.DataFrame.from_dict(request, orient='index').T
    score = model.predict_proba(dataframe)[0][1]
    
    return score


app = FastAPI(title='patient-api', version='0.1')

@app.post('/check_patient/', response_model=ResponseModel)
def check_patient(patient: Patient):
    start_time = datetime.now()

    score = get_score(patient.model_dump())
    class_ = 'positive' if score > 0.5 else 'negative'

    runtime = datetime.now() - start_time

    return ResponseModel(score=score, class_=class_, runtime=runtime)

if __name__ == '__main__':
    uvicorn.run("ugly_api:app", host='127.0.0.1', port=5000)

