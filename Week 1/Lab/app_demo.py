import shap
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostClassifier


df = pq.read_table('artifacts/demo_data.pq').to_pandas() #1

if 'patient_list' not in st.session_state:
    st.session_state.patient_list = set(df['patient_nbr'].sample(100))

if 'show_raw_data' not in st.session_state:
    st.session_state.show_raw_data = False

st.title('Readmission Risk')

st.set_option('deprecation.showPyplotGlobalUse', False) #2

option= st.selectbox('Select a patient ID', st.session_state.patient_list)


def load_model():
    model= CatBoostClassifier()
    model.load_model('artifacts/scoring_model.cbm')
    
    return model


def get_features(pt_id, model, age, df=df):
    dataframe= df[df['patient_nbr'] == pt_id][model.feature_names_]
    dataframe['age'] = age
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
    model = load_model()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    return explainer


def get_shap_values(requested_data):
    explainer = get_explainer()
    shap_values = explainer(requested_data)
    
    return shap_values
   

def get_waterfall_plot(shap_values):
    plt.figure(figsize=(10,10))
    fig = shap.plots.waterfall(shap_values[0], max_display=10)
    return fig
    

modelx = load_model()
initial_age = df[df['patient_nbr'] == option]['age'].values[0]
age = st.slider('Select an age:', min_value=15, max_value=100, value=initial_age)
features, score, dfx = get_features(option, modelx, age)
shap_values = get_shap_values(dfx)
graph = get_waterfall_plot(shap_values)

st.header('Risk for the patient')
# st.write(score)
col1, col2, col3 = st.columns(3)
col2.write(score)
col2.markdown(get_style(score), unsafe_allow_html=True)

st.header('Main drivers of the decision')
st.pyplot(graph)

st.header('Features for the patient')

if st.button('Toggle Raw Data'):
    st.session_state.show_raw_data = not st.session_state.show_raw_data

if st.session_state.show_raw_data:
    st.write(features)