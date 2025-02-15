import pandas as pd 
import numpy as np
import time
import glob 
import os 
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.optimizers import Adam

from keras.models import load_model

import pandasql as ps
from pandasql import sqldf

import streamlit as st

import Param as config

def parse_date(row):
    annee = int(row['an'])
    mois = int(row['mois'])
    jour = int(row['jour'])
    hrmn = row['hrmn']
    
    splitted= hrmn.split(':')
    heure = int(splitted[0])
    mn = int(splitted[1])

    return pd.Timestamp(annee, mois, jour, heure, mn)

# Function to create sequences for training the LSTM
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def create_future_sequences(data, seq_length, num_future_steps):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences[-num_future_steps:])


def predict(n_days):

    path = config.PATH
    path_csv_out = path + 'csv_out'

    df_accidents = pd.read_csv(os.path.join(path_csv_out,"accidents.csv"),sep=';',low_memory=False)


    # Apply the custom function to create a 'date' column
    df_accidents['date'] = df_accidents.apply(parse_date, axis=1)

    df_accidents = df_accidents.drop(['an','mois','jour','hrmn'],axis=1).copy()

    df_accidents['date'] = pd.to_datetime(df_accidents['date'])
    df_accidents['date'] = df_accidents['date'].dt.strftime('%Y-%m-%d')

    columns = ['date','grav']
    df_accidents = df_accidents[columns].copy()

    df_casualty = sqldf('''SELECT date,count(*) as n_casualty FROM df_accidents GROUP BY date''')
    df_casualty['date'] = pd.to_datetime(df_casualty['date'])

    df_casualty = df_casualty.reset_index()

    model = load_model("model_lstm.keras")

    df=df_casualty.copy()
    #df = df.sort_values('date')
    data = df['n_casualty'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

    sequence_length = 10

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    n_timesteps = X_train.shape[1]
    n_features = 1

    X_future = create_future_sequences(data_scaled, sequence_length,n_days)
    X_future = X_future.reshape(X_future.shape[0], X_future.shape[1], 1)
    future_predictions = model.predict(X_future)
    future_predictions = scaler.inverse_transform(future_predictions)

    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=len(future_predictions), freq='D')
    future_df = pd.DataFrame({'date': future_dates, 'predicted_n_casualty': future_predictions.flatten()})

    fig = plt.figure(figsize=(20, 10))
    plt.plot(df['date'], df['n_casualty'], label="Valeurs réelles sur le dataset complet")
    plt.plot(future_df['date'], future_df['predicted_n_casualty'], label='Prédictions futures', linestyle='dashed', color='green')
    plt.legend()
    plt.title("Prediction pour les "+str(n_days)+" prochains jours")
    plt.xlabel('Temps')
    plt.ylabel("Nombre d'accidents")
    st.pyplot(fig)

    return