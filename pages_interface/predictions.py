# -*- coding: utf-8 -*-
"""
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_featuring import create_datetime_features


def mpl_display_ts_multiple_predictions(df_data, l_pred_col, true_y, x_label='', y_label='',
                                        title="Power consumption prediction",
                                        ax_titles=None):
    
    n_pred = len(l_pred_col)
    fig, ax = plt.subplots(n_pred, 1, figsize=(10, 8), constrained_layout=True,
                           sharex=True, sharey=True)
    
    if ax_titles is None or len(ax_titles) < n_pred:
        ax_titles=['Prediction {}'.format(i) for i in range(n_pred)]

    for i in range(n_pred):
        df_data.plot(y=true_y, ax=ax[i], color='k', linewidth=1)
        df_data.plot(y=l_pred_col[i], ax=ax[i], color='b', 
                             style='.',linewidth=0.1, alpha=0.2)
        ax[i].set_title(ax_titles[i])
        ax[i].legend(["Raw Data", l_pred_col[i]], bbox_to_anchor=(1.02, 1), 
                     loc='upper left', borderaxespad=0)
        ax[i].set_ylabel(y_label)
    fig.suptitle(title)
    return fig

def sns_display_boxplot(df_data, x, l_pred_col, true_y, x_label='', y_label='',
                        title="Power consumption prediction",
                        ax_titles=None):
    
    palettes = ['Reds', "Blues", "Greens", "Oranges"]
    n_pred = len(l_pred_col)
    if ax_titles is None or len(ax_titles) < n_pred:
        ax_titles=['Prediction {}'.format(i) for i in range(n_pred)]
    
    fig, ax = plt.subplots(n_pred+1, 1, figsize=(10, 8), constrained_layout=True,
                           sharex=True, sharey=True)
    sns.boxplot(data=df_data, x=x, ax=ax[0], y=true_y, palette=palettes[0])
    ax[0].set_xlabel('')
    for i in range(n_pred):
        sns.boxplot(data=df_data, x=x, ax=ax[i+1],
                    y=l_pred_col[i], palette=palettes[i+1])
        ax[i+1].set_title(ax_titles[i])
        ax[i+1].set_ylabel(y_label)
        ax[i+1].set_xlabel('')
        
    ax[-1].set_xlabel(x_label)        
    ax[0].set_title("Données mesurées")
    ax[0].set_ylabel(y_label)
    fig.suptitle(title)
    return fig


def read_pred_results(file_pkl):
    with open(file_pkl, 'rb') as fp:
        df_predictions = pickle.load(fp)
    return df_predictions
    
def predictions():
    # Différents fichiers de résultats
    # Zone 1
    file_z1_xgboost = "predictions/Zone1/xgboost_predictions_Consumption_Z1.pkl"
    file_z1_lstm1h = "predictions/Zone1/lstm1H_predictions_Consumption_Z1.pkl"
    file_z1_lstm = "predictions/Zone1/lstm_predictions_Consumption_Z1.pkl"
    # Zone 2
    file_z2_xgboost = "predictions/Zone2/xgboost_predictions_Consumption_Z2.pkl"
    file_z2_lstm1h = "predictions/Zone2/lstm1H_predictions_Consumption_Z2.pkl"
    file_z2_lstm = "predictions/Zone2/lstm_predictions_Consumption_Z2.pkl"
    # Sone 3
    file_z3_xgboost = "predictions/Zone3/xgboost_predictions_Consumption_Z3.pkl"
    file_z3_lstm1h = "predictions/Zone3/lstm1H_predictions_Consumption_Z3.pkl"
    file_z3_lstm = "predictions/Zone3/lstm_predictions_Consumption_Z3.pkl"
    
    # chargement des différentes données
    df_z1_xgboost = read_pred_results(file_z1_xgboost)
    df_z1_lstm1h = read_pred_results(file_z1_lstm1h)
    df_z1_lstm = read_pred_results(file_z1_lstm)
    
    df_z2_xgboost = read_pred_results(file_z2_xgboost)
    df_z2_lstm1h = read_pred_results(file_z2_lstm1h)
    df_z2_lstm = read_pred_results(file_z2_lstm)
    
    df_z3_xgboost = read_pred_results(file_z3_xgboost)
    df_z3_lstm1h = read_pred_results(file_z3_lstm1h)
    df_z3_lstm = read_pred_results(file_z3_lstm)
    
    l_pred_col=["XGBoost", "LSTM_1H", "LSTM_1H_1variable"]
    ax_titles=["Prédictions XGBoost", "Prédictions LSTM_1H",
                        "Prédictions LSTM_1H_1variable"]
    
    st.write("# PowerForecast")
    with st.container():
       st.write("")
       st.write(f""" ### Comparaison des différents modèles  """)
       st.info("""
          :point_down: **Résultats obtenus par zone**
      """)
      
    with st.container():
        zone_col, window_col, kind_col = st.columns(3)
        
        with zone_col:
            zones = st.radio('**Zones de la ville de Tétouan**',
                             ('Zone 1', 'Zone 2', 'Zone 3'))
        
        with kind_col:
              kind = st.radio("**Type de la visualisation**",
                              ('Temporelle', 'Autre'), index=1)    
        with window_col:
            window = st.radio("**Fenêtre d'analyse**",
                              ('Globale', 'Personnalisée'))
     
    if zones=="Zone 1":
        # Plage de dates des données
        first_day = np.array([df_z1_xgboost.index[0], df_z1_lstm.index[0],
                              df_z1_lstm1h.index[0]]).max()
        last_day = np.array([df_z1_xgboost.index[-1], df_z1_lstm.index[-1],
                             df_z1_lstm1h.index[-1]]).min()
        
        # Merge the dataframe
        df_z1_xgboost.rename(columns={"prediction":"XGBoost"}, inplace=True)
        df_z1_lstm1h.rename(columns={"prediction":"LSTM_1H"}, inplace=True)
        df_z1_lstm.rename(columns={"prediction":"LSTM_1H_1variable"}, inplace=True)
        df_data = pd.concat([df_z1_xgboost, df_z1_lstm1h[["LSTM_1H"]],
                             df_z1_lstm[["LSTM_1H_1variable"]]], axis=1)
        target="Consumption_Z1"
    if zones=="Zone 2":
        # Plage de dates des données
        first_day = np.array([df_z2_xgboost.index[0], df_z2_lstm.index[0],
                              df_z2_lstm1h.index[0]]).max()
        last_day = np.array([df_z2_xgboost.index[-1], df_z2_lstm.index[-1],
                             df_z2_lstm1h.index[-1]]).min()
        
        # Merge the dataframe
        df_z2_xgboost.rename(columns={"prediction":"XGBoost"}, inplace=True)
        df_z2_lstm1h.rename(columns={"prediction":"LSTM_1H"}, inplace=True)
        df_z2_lstm.rename(columns={"prediction":"LSTM_1H_1variable"}, inplace=True)
        df_data = pd.concat([df_z2_xgboost, df_z2_lstm1h[["LSTM_1H"]],
                             df_z2_lstm[["LSTM_1H_1variable"]]], axis=1)
        target="Consumption_Z2"
        
    if zones=="Zone 3":
        # Plage de dates des données
        first_day = np.array([df_z3_xgboost.index[0], df_z3_lstm.index[0],
                              df_z3_lstm1h.index[0]]).max()
        last_day = np.array([df_z3_xgboost.index[-1], df_z3_lstm.index[-1],
                             df_z3_lstm1h.index[-1]]).min()
        
        # Merge the dataframe
        df_z3_xgboost.rename(columns={"prediction":"XGBoost"}, inplace=True)
        df_z3_lstm1h.rename(columns={"prediction":"LSTM_1H"}, inplace=True)
        df_z3_lstm.rename(columns={"prediction":"LSTM_1H_1variable"}, inplace=True)
        df_data = pd.concat([df_z3_xgboost, df_z3_lstm1h[["LSTM_1H"]],
                             df_z3_lstm[["LSTM_1H_1variable"]]], axis=1)
        target="Consumption_Z3"
        
    if window == 'Globale':           
        with st.container():
            if kind == 'Temporelle':
                with st.spinner(text="En cours ..."):
                    st.pyplot(mpl_display_ts_multiple_predictions(
                        df_data, l_pred_col=l_pred_col,
                        true_y=target, x_label='', y_label="Consumption (KW)",
                        title="", ax_titles=ax_titles))
            else:
                df_data = create_datetime_features(df_data=df_data)
                with st.container():
                    st.write("**Consommation en KW par mois**")
                    st.pyplot(sns_display_boxplot(df_data, x="Month", l_pred_col=l_pred_col,
                                                  true_y=target, y_label='',
                                                  title="",
                                                  ax_titles=ax_titles,
                                                  x_label="Numéro de mois de l'année"))
                with st.container():
                    st.write("**Consommation en KW par heure**")
                    st.pyplot(sns_display_boxplot(df_data, x="Hour", l_pred_col=l_pred_col,
                                                  true_y=target, y_label='',
                                                  title="",
                                                  ax_titles=ax_titles,
                                                  x_label="Heures de la journée"))
    else:
       # Sélectionner un intervalle de dates
       st.info("""
           :point_down: **Période de visualisation personnalisée.**
       """)
       date_range = \
           st.date_input("Choisisez une période de visualisation",
                         value=(first_day,
                                first_day+pd.DateOffset(days=7)),
                         min_value=first_day, max_value=last_day)
       if len(date_range) == 1:
           date_range = (date_range[0], 
                         min(date_range[0]+pd.DateOffset(days=7), last_day))
       
       if kind == 'Temporelle':
           df_sub_data = df_data[date_range[0]:date_range[1]]
           with st.spinner(text="En cours ..."):
               st.pyplot(mpl_display_ts_multiple_predictions(
                   df_sub_data, l_pred_col=l_pred_col,
                   true_y=target, x_label='', y_label="Consumption (KW)",
                   title="", ax_titles=ax_titles))
       else:
           df_data = create_datetime_features(df_data=df_data)
           df_sub_data = df_data[date_range[0]:date_range[1]]
           with st.container():
               st.write("**Consommation en KW par heure**")
               st.pyplot(sns_display_boxplot(df_sub_data, x="Hour", l_pred_col=l_pred_col,
                                             true_y=target, y_label='',
                                             title="",
                                             ax_titles=ax_titles,
                                             x_label="Heures de la journée"))
