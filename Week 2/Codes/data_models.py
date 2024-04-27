from enum     import Enum
from typing   import Optional
from pydantic import BaseModel

class Race(str,Enum): #1
    AfricanAmerican = 'AfricanAmerican'
    Caucasian= 'Caucasian'
    Other    = 'Other'
    Asian    = 'Asian'
    Hispanic = 'Hispanic'

class Patient(BaseModel): #1
    gender: str
    age: int
    weight: str
    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: float
    diag_2: float
    diag_3: float
    number_diagnoses: int
    max_glu_serum: int
    a1cresult: int
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesmed: str
    race: Optional[Race] = Race.Other #2
    
class Response(BaseModel): #3
    Score  : float
    Class  : int
    runtime: float