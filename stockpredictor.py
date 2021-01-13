import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from PIL import Image
import math
from sklearn.metrics import mean_squared_error


import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler

def run_app():
    image=Image.open('Bull_vs_bear.png')
    st.image(image,use_column_width=True)

    #file_uploader=st.file_uploader("Upload NCDEX stock market file for prediction",type=["csv"])
    st.sidebar.info("This application focuses on forecasting closing price of commodities by using Long  Short term memory Model")

    st.sidebar.info("Application Developed by Siddhesh D. Munagekar for predicting multiple Commodity closing Price")
    #if file_uploader is not None:

    ####Uploading NCEX Commodity exchange Dataset#############

    url = 'https://drive.google.com/file/d/18i0ledDo3NpT8Km03y32a_-nbvlDrr6t/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

    df=pd.read_csv(path)
    df=df[['Commodity','Closing Price']]
    df = df.dropna(subset=['Closing Price'])
    commodity_list=df['Commodity'].unique().tolist()
    commodity=st.sidebar.selectbox("Select Commodity",commodity_list)
    df1 = df[df['Commodity'] == commodity].reset_index(drop=True)['Closing Price']

    fig = plt.figure()
    plt.plot(df1)
    plt.xlabel(commodity)
    plt.ylabel('Closing Price')
    st.pyplot(fig)
    if st.checkbox("Would like to view  Closing price of  "+commodity):
        st.write(df1)
    st.write("Total number of rows fetched corresponding to "+commodity +" :  {}".format(len(df1)))

    #Scaling the data

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    ##Preparing the size of train set and test set
    training_size = int(len(df1) * 0.75)
    test_size = len(df1) - training_size
    train_set, test_set = df1[0:training_size, :], df1[training_size:len(df1), :1]

        #Splitting the data into train set and test set by shifting it according to time step
    def create_dataset(dataset, time_step):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    st.sidebar.info("Time Step is the total number of previous prices as an input based on which output is predicted")
    time_step = st.sidebar.slider('Time step', min_value=0, max_value=10, value=5)

    X_train, y_train = create_dataset(train_set, time_step)
    X_test, y_test = create_dataset(test_set, time_step)

        # reshape input as [samples,time_step,feature] convert it to 3 dimensional array as required by LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Creating a stacked LSTM model


    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    #Training a model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

        #Calculating Mean squared error
    train_mse=math.sqrt(mean_squared_error(y_train, train_predict))
    test_mse=math.sqrt(mean_squared_error(y_test, test_predict))
    st.write("Test mean squared Error",test_mse)
    st.write("Train mean squared Error",train_mse)

        #Considering last 10 days output
    x_input = test_set[(len(test_set) - time_step):].reshape(1, -1)
    # Converting it to list
    temp_input = list(x_input)
    # Arranging list vertically
    temp_input = temp_input[0].tolist()

    st.sidebar.info("Future Days are the number of days based on which closing price will be forecasted")

    futr_days = st.sidebar.number_input('Select future days', min_value=0, max_value=10, value=5)

    # demonstrate prediction for next 10 days

    lst_output = []

    future_day = futr_days
    n_steps = time_step
    i = 0
    # Forcast next 10 days output
    while (i < future_day):

        if (len(temp_input) > n_steps):

            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            # Converting to 3d array for lstm
            x_input = x_input.reshape(1, n_steps, 1)
            # print(x_input)
            ypred = model.predict(x_input, verbose=0)
            #print("{} day predicted output {}".format(i, ypred))
            # adding predicted output  to temp_input list
            temp_input.extend(ypred[0].tolist())
            temp_input = temp_input[1:]

                # print(temp_input)
            lst_output.extend(ypred.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            ypred = model.predict(x_input, verbose=0)
            print("Predicted y of 0 day", ypred[0])
            # Addding ypred value in temp_input(previous input)
            temp_input.extend(ypred[0].tolist())
            #print(len(temp_input))
            lst_output.extend(ypred.tolist())
            i = i + 1

        #print(lst_output)

    previous_days = np.arange(len(df1) - n_steps, len(df1))
    predicted_future = np.arange(len(df1), len(df1) + future_day)




    # Selecting last 10 days input from the dataframe df1 for first plot
    fig2 = plt.figure()
    plt.title("Forecasted prices of "+ commodity + " for next "+str(future_day) + " days")
    plt.xlabel("Future Days")
    plt.ylabel("Closing Price")
    plt.plot(previous_days, scaler.inverse_transform(df1[len(df1) - n_steps:]))
    # Selecting predicted output from the list of the above function
    plt.plot(predicted_future, scaler.inverse_transform(lst_output))

    st.pyplot(fig2)


    outputlist = df1.tolist()
    outputlist.extend(lst_output)
    if st.checkbox("Would you like to view combined original and forecasted graph"):
        fig1=plt.figure()
        plt.title("Combined graph with forecasted price")
        plt.xlabel("Future Days")
        plt.ylabel("Closing Price")
        plt.plot(np.append(previous_days, predicted_future),scaler.inverse_transform(outputlist[len(df1) - n_steps:]))
        plt.plot(predicted_future, scaler.inverse_transform(lst_output))
        st.pyplot(fig1)

if __name__=='__main__':
    run_app()


