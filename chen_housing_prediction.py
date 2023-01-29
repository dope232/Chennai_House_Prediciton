import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor


st.write("""
# Chennai House Prediction App 
Predict House prices in chennai """)

st.write("---")

city = pd.read_csv("C:/Users/dhanu/tk/clean_data.csv")
city = city.dropna()
st.dataframe(city)

X1 = city.price
Y1 = city.area
Y2 = city.bathroom
Y3 = city.bhk
Y4 = city.age


plot = px.scatter(data_frame=city, x = X1, y = Y1 )
plot2 = px.scatter(data_frame=city, x = X1, y = Y2 )
plot3 = px.scatter(data_frame=city, x = X1, y = Y3 )
plot4 = px.scatter(data_frame=city, x = X1, y = Y4 )



st.plotly_chart(plot)
st.plotly_chart(plot2)
st.plotly_chart(plot3)
st.plotly_chart(plot4)

st.header("Specify Input Parameters")
def user_input_features():
     bed = st.number_input('Insert  number of bedrooms', key = 'bed')
     bath = st.number_input('Insert a number of bathrooms', key = 'bath')
     area = st.number_input('Insert a number of sqft area', key = 'area')
     status = st.number_input('Insert a number 1 = constructed, 0 = not constructed', key = 'status')
     location = st.number_input('Insert a number, any number for location- Location key in next page ', key = 'location')


     data = {'area': area,
             'status': status,
             'bed' : bed,

             'bath': bath,
             'location': location,

             }
     features = pd.DataFrame(data, index=[0])
     return features

df = user_input_features()

reg = LinearRegression()
le = LabelEncoder()

labels = city['price']
conv_status = [1 if values == "Ready to move" else 0 for values in city.status]
city['status'] = conv_status
city['location'] = le.fit_transform(city['location'])

train1 = city.drop(['builder','age'], axis = 1)

st.dataframe(train1)


x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size= 0.60, random_state=9)

reg.fit(x_train,y_train)




clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')

clf.fit(x_train,y_train)
st.header("Prediction of this model")
st.header(clf.score(x_test,y_test))

st.write(" Which is",(clf.score(x_test,y_test)*100), "%")

Y = y_test

X = x_test.drop('price', axis = 1)


reg.fit(X,Y)
prediction3 = reg.predict(df)


st.header('Specified Input parameters')
st.write(df)


st.header("Thus the predicted house price is ")
st.write(prediction3)










