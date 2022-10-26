# module_13 - Venture Funding with Deep Learning

The purpose of this project is to create a model that predicts whether applicants for funding will be successful if funded.

---
## Technologies

* python 3.9
* TensorFlow 
* Scikit learn

---
## Imports and Dependencies

* import pandas as pd
* from pathlib import Path
* import tensorflow as tf
* from tensorflow.keras.layers import Dense
* from tensorflow.keras.models import Sequential
* from sklearn.model_selection import train_test_split
* from sklearn.preprocessing import StandardScaler,OneHotEncoder

---
## Usage

After loading the csv file and dropping columns that are not applicable the data is prepared by 

creating a list of categorical variables and 
'categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index)'

encode the categorcal variables using OneHotEncoder.
'''enc = OneHotEncoder(sparse=False)
encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])'''

And create a new dataframe with the OneHot data and concat with the applicant_data_df
'encoded_df = pd.concat([numerical_variables_df, encoded_df], axis=1, join='inner')'

Now Compile and Evaluate a Binary Classification Model Using a Neural Network

Then compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric and optimize 

---
## Contributors

Ryan Svendson
rsvendson@gmail.com