import re

from datetime import datetime
from utils import data_reader,check_nulls, age_dict, \
max_glu_serum_dict, a1cresult_dict,load_model,logger, \
impute_race_column, train

COLUMNS= ['race','gender','age','weight','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital',
     'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient',
     'diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','a1cresult','metformin','repaglinide','nateglinide',
     'chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone',
     'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide_metformin',
     'glipizide_metformin','glimepiride_pioglitazone','metformin_rosiglitazone','metformin_pioglitazone',
     'change','diabetesmed','encounter_id', 'patient_nbr','readmitted']

def train_model():
    #Read Data
    df= data_reader(path= './artifacts/diabetic_data_structured.pq',
                    cols= COLUMNS)
    #Check Nulls
    check_nulls(df)
    
    #Manipulate features
    df.age = df.age.map(age_dict)
    df.max_glu_serum = df.max_glu_serum.map(max_glu_serum_dict)
    df.a1cresult = df.a1cresult.map(a1cresult_dict)
    
    logger.info(f'Max glu serum value counts:\n{df.max_glu_serum.value_counts()}')
    logger.info(f'A1cresult valuecounts:\n{df.a1cresult.value_counts()}')

    #Impute Race Column
    df= impute_race_column(df)

    #Train Test Split and Pool
    target = 'readmitted'
    not_input = ['readmitted', 'patient_nbr', 'encounter_id']
    input_features = [column for column in df.columns if column not in not_input]
    cat_features= list(df[input_features].select_dtypes(include='object').columns)

    #Manipulate diag columns
    df[['diag_1', 'diag_2', 'diag_3']] = df[['diag_1', 'diag_2', 'diag_3']].applymap(lambda x: float(re.sub('\D', '', x)) 
                                                                                 if isinstance(x, str) else x)
    #Change datatype of ids
    df[['admission_type_id',
        'discharge_disposition_id',
        'admission_source_id']]= df[['admission_type_id',
                                     'discharge_disposition_id',
                                     'admission_source_id']].astype('str')

    model=train(df)
    model.save_model(f'artifacts/diabetes_model_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.cbm')

if __name__ == '__main__':
    train_model()