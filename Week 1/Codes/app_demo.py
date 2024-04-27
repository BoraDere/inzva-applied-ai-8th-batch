import shap
import pandas as pd
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostClassifier

#---------------------------------------------------------------------------------------------#
#-----------------------------******Initialization*****---------------------------------------#
#---------------------------------------------------------------------------------------------#

# Load data to show
df          = pq.read_table('artifacts/demo_data.pq').to_pandas() #1
patient_list= set(df['patient_nbr'].head(100)) #1
    
    
#Title element
st.title('Readmission Risk')

st.set_option('deprecation.showPyplotGlobalUse', False) #2

option= st.selectbox('Select a patient ID',patient_list)

#---------------------------------------------------------------------------------------------#
#-----------------------------User Defined Functions***---------------------------------------#
#---------------------------------------------------------------------------------------------#

#Load model function
def load_model():
    
    model= CatBoostClassifier()
    model.load_model('artifacts/scoring_model.cbm')
    
    return model

#func for getting selected patient features and score
def get_features(pt_id, model,df=df):
    
    dataframe= df[df['patient_nbr'] == pt_id][model.feature_names_]
    features = dataframe.to_dict(orient= 'records')[0]
    score    = model.predict_proba(dataframe)[0][1]
    
    return features, score, dataframe

def get_style(score):
    
    if score > 0.5:
        
        text= f'<p style="font-family:Georgia; color:Red; font-size: 30px;"> % {str(np.round(score*100,2))}</p>' #3  
        
    else:
        text= f'<p style="font-family:Georgia; color:Green; font-size: 30px;"> % {str(np.round(score*100,2))}</p>' #3        
    
    return text

def get_explainer():
    
    model= load_model()
    explainer   = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    return explainer

def get_shap_values(requested_data):
    
    explainer= get_explainer()
    shap_values = explainer(requested_data)
    
    return shap_values
   
def get_waterfall_plot(shap_values):
    
    plt.figure(figsize=(10,10))
    fig= shap.plots.waterfall(shap_values[0], max_display=10)
    return fig
    
#---------------------------------------------------------------------------------------------#
#-----------------------------Define Variables***---------------------------------------#
#---------------------------------------------------------------------------------------------#


modelx               = load_model()
features, score, dfx = get_features(option,modelx)
shap_values          = get_shap_values(dfx)
graph                = get_waterfall_plot(shap_values)

#---------------------------------------------------------------------------------------------#
#-----------------------------Design the App***-----------------------------------------------#
#---------------------------------------------------------------------------------------------#

st.header('Risk for the patient')
st.write(score)
#col1,col2,col3= st.columns(3)
#col2.write(score)
#col2.markdown(get_style(score), unsafe_allow_html= True)

st.header('Main drivers of the decision')
st.pyplot(graph)

st.header('Features for the patient')
st.write(features)