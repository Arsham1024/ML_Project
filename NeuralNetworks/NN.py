import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as pre
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import *


# Categorical data
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina" , "ST_Slope"]

# load data training and testing is combined here.
db = pd.read_csv("./input_data/heart.csv")
print(db.info() , '\n')

# encoding the categorical data into numerics so we can use them to train the model.
for col in categorical:
    db[col].unique()

    # label_encoder object knows how to understand word labels.
    label_encoder = pre.LabelEncoder()
    # fit and transform data from the original set and put it back
    db[col] = label_encoder.fit_transform(db[col])

    db[col].unique()

print(db)

# # Dropping the cols with categorical value initially to make the model
# db = db.drop(columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina" , "ST_Slope"])
# print(db.head())

# shape X and y
# db.loc[rows , cols by lables]
X , y = db.loc[:, 'Age':'Oldpeak'] , db.loc[ : , 'HeartDisease']
X=np.asarray(X).astype(np.float32)
y=np.asarray(y).astype(np.float32)

# making test and train samples
# X_train includes 642 rows
# X_test includes 276
X_train , X_test, Y_train, Y_test = train_test_split(X , y , test_size=0.3 , random_state=30)

# making the model
model = Sequential()
model.add(Dense(11 ,input_dim=10, activation='relu'))
model.add(Dense(20 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))
model.add(Dense(21 ,activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse' , optimizer="adam")


# training
model.fit(
    X_train,
    Y_train,
    epochs= 1175,
    shuffle=True,
    verbose=2
)

# Error rate
error = model.evaluate(X_test, Y_test , verbose=0)

print(f"error rate is : {(1-error)*100}%")

# Save the nural network
# .h5 aka htf5 format is a binary file format for python array data
model.save("./output_data/NN_model.h5")




