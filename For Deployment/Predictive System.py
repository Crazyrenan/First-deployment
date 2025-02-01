# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:14:41 2024

@author: Samuel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:57:03 2024

@author: Samuel
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users\Samuel/Anaconda Tasks/Data Modelling/Final/trained_model.sav', 'rb'))

# 0 = High, 1 = Low, 2 = Normal

input_data = (4, 5, 4, 3, 2, 2, 3, 4, 4, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('High')
elif prediction[0] == 1:
    print('Low')
else:
    print('Normal')