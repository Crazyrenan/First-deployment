# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:25:55 2024

@author: Samuel
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users\Samuel/Anaconda Tasks/Data Modelling/Final/trained_model.sav', 'rb'))


def lung_cancer_prediction(input_data):
    # sleep apnea = 1, insomnia = 0
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print('Test Input :', input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'High'
    elif prediction[0] == 1:
        return 'Low'
    else:
        return 'Normal'
def main():
    
    st.title('Lung Cancer Web App')
    Alcohol_Use = st.number_input('Alcohol Use')
    Dust_Allergy = st.number_input('User Dust Allergy')
    OccuPational_Hazards = st.number_input('User OccuPational Hazards')
    Genetic_Risk = st.number_input('User Genetic Risk')
    chronic_Lung_Disease = st.number_input('User chronic Lung Disease')
    Balanced_Diet = st.number_input('User Balanced Diet')
    Smoking = st.number_input('User Smoking')
    Coughing_of_Blood = st.number_input('User Coughing of Blood')
    Obesity = st.number_input('User Obesity')
    Air_Pollution = st.number_input('User Air Pollution')
    Passive_Smoker = st.number_input('User Passive Smoker')

    
    diagnosis = ''
    
    if st.button('Lung Cancer Test result'):
        diagnosis= lung_cancer_prediction([Alcohol_Use,Dust_Allergy,OccuPational_Hazards,Genetic_Risk,
                                             chronic_Lung_Disease,Balanced_Diet,Smoking,
                                             Coughing_of_Blood,Obesity,Air_Pollution,Passive_Smoker])
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    

