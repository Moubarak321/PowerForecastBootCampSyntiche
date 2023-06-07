# -*- coding: utf-8 -*-
"""
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime

from src.data_gathering import load_dataset
from src.data_analysis import apply_sanity_check
from src.data_featuring import create_datetime_features

def mpl_display_ts(dataset, y, x_label='',
                   title="Power consumption"):
    fig, ax = plt.subplots(figsize=(10, 8))
    dataset[y].plot(y=y, ax=ax)
    ax.set_ylabel(y, fontsize=14)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return fig

def sns_display_boxplot(dataset, x, y, x_label = '',
                        title="Power consumption",
                        palette="Greens"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=dataset, x=x, ax=ax,
                      y=y, palette=palette)
    ax.set_ylabel(y, fontsize=14)
    ax.set_xlabel(x_label, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(title)
    return fig

def data_analysis():
    with st.container():
        st.write("# PowerForecast")
    with st.container():
        st.write("")
        st.write(f""" ### Visualisation des données    """)

        st.info("""
            :point_down: **Les données peuvent être visualisées de plusieurs façons afin d'enrichir l'analyse.**
        """)
       
        st.write(f"""
        - **Temporelle** : permet d'afficher les séries temporelles de la fenêtre d'analyse chosiie
        - **Autres**: Permet de visualuser la répartition des consommations mensuelles et journalières. C'est une approche qui permet une analyse globale pertinente. 
        """)
        
        st.info("""
            :point_down: **Pour permettre une analyse fine, plusieurs niveaux de granularité de l'analyse sont proposés.**
        """)
        
        st.write(f"""
            - **Globale** : toutes les données disponibles sont agrégées.
            - **Mensuelle** : les donénes peuvent être analysées par numéro de mois. 
            - Personnalisée : la période d'analyse est libre. Elle peut être personalisée grâce au calendrier disponbile. 
            Par défaut une période de 30 jours est choisie à partir du premiere jour.
        """)

    # Charger les données à partir du fichier csv
    data_filename = "data/PowerConsumptionTetouan/Tetuan_City_power_consumption.csv"
    df_dataset = load_dataset(data_filename, col_sep=",", b_rename_cols=True)

    # Effectuer la vérification des données
    #d_sanity_check = apply_sanity_check(df_dataset)
    #df_sanity_check = pd.DataFrame(d_sanity_check)
    
    # Enrichir les données avec des informations complémetaires sur le temps
    df_dataset = create_datetime_features(df_data=df_dataset)
    
    # Plage de dates des données
    first_day = df_dataset.index[0]
    last_day = df_dataset.index[-1]
    
    st.write(f""" ####  Analyse des données de la ville de Tétouan""")
      
    
    with st.container():
        zone_col, window_col, kind_col = st.columns(3)
        
        with zone_col:
            zones = st.multiselect(
        '**Zones de la ville de Tétouan**',
        ['Zone 1', 'Zone 2', 'Zone 3'], ['Zone 1', 'Zone 2', 'Zone 3'],
        )
        
        with kind_col:
            kind = st.radio("**Type de la visualisation**",
                            ('Temporelle', 'Autre'), index=1)
            
        with window_col:
            window = st.radio(
            "**Fenêtre d'analyse**",
            ('Globale', 'Mensuelle', 'Personnalisée'))
        
    if kind == 'Temporelle':  
        if window == 'Globale':
            with st.container():
                if "Zone 1" in zones:
                    with st.container():
                        st.pyplot(mpl_display_ts(df_dataset, y='Consumption_Z1',
                                                 title="Zone 1 power consumption (KW)"))
                if "Zone 2" in zones:
                    with st.container():
                        st.pyplot(mpl_display_ts(df_dataset, 'Consumption_Z2',
                                                 title="Zone 2 power consumption (KW)"))
                if "Zone 3" in zones:
                    with st.container():
                        st.pyplot(mpl_display_ts(df_dataset, 'Consumption_Z3',
                                                 title="Zone 3 power consumption (KW)"))
        
        elif window == 'Mensuelle':
            rand_month = int(random.randint(1, 12))
            number = st.number_input('Indiquer un numéro de mois (de 1 à 12)',
                                     step=1, min_value=1, max_value=12,
                                     format="%d", value=rand_month)
            if (number < 1) or (number > 12):
                number = rand_month
            df_sub_data = df_dataset[df_dataset['Month']==number]
            with st.container():
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z1',
                                         title="Zone 1 power consumption (KW)"))
            with st.container():
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z2',
                                         title="Zone 2 power consumption (KW)"))
            with st.container():
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z3',
                                         title="Zone 3 power consumption (KW)"))
                
        elif window == 'Personnalisée':
            # Sélectionner un intervalle de dates
            date_range = \
                st.date_input("Choisisez une période de visualisation",
                              value=(first_day,
                                     first_day+pd.DateOffset(days=30)),
                              min_value=first_day, max_value=last_day)
            
            if len(date_range) == 1:
                date_range = (date_range[0], 
                              date_range[0]+pd.DateOffset(days=30))
            
            df_sub_data = df_dataset[date_range[0]:date_range[1]]
            with st.container():
                st.write('**Zone 1 consumption power (KW)**')
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z1'))
            with st.container():
                st.write('**Zone 2 consumption power (KW)**')
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z2'))
            with st.container():
                st.write('**Zone 3 consumption power (KW)**')
                st.pyplot(mpl_display_ts(df_sub_data, 'Consumption_Z3'))

    else:
        if window == 'Globale':
            with st.container():
                st.write("**Zone 1 : consommation en KW par mois et par heure**")
                month_col, hour_col = st.columns(2)
                with month_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Month',
                                                  y='Consumption_Z1',
                                                  palette='Reds',
                                                  x_label="Mois de l'année",
                                                  title="Consommation mensuelle de la Zone 1 (KW)"))
                with hour_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Hour',
                                                  y='Consumption_Z1',
                                                  palette='Reds',
                                                  x_label="Heures de la journée",
                                                  title="Consommation mensuelle de la Zone 1 (KW)"))
            with st.container():
                st.write("**Zone 2 : consommation en KW par mois et par heure**")
                month_col, hour_col = st.columns(2)
                with month_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Month',
                                                  y='Consumption_Z2',
                                                  x_label="Mois de l'année",
                                                  palette='Blues',
                                                  title=""))
                with hour_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Hour',
                                                  y='Consumption_Z2',
                                                  x_label="Heures de la journée",
                                                  palette='Blues',
                                                  title=""))
            with st.container():
                st.write("**Zone 3 : consommation en KW par mois et par heure**")
                month_col, hour_col = st.columns(2)
                with month_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Month',
                                                  x_label="Mois de l'année",
                                                  y='Consumption_Z3',
                                                  palette='Greens',
                                                  title=""))
                with hour_col:
                    st.pyplot(sns_display_boxplot(df_dataset, x='Hour',
                                                  x_label="Heures de la journée",
                                                  y='Consumption_Z3',
                                                  palette='Greens',
                                                  title=""))
                
        elif window == 'Mensuelle':
            rand_month = int(random.randint(1, 12))
            number = st.number_input('Indiquer un numéro de mois (de 1 à 12)',
                                     step=1, min_value=1, max_value=12,
                                     format="%d", value=rand_month)
            if (number < 1) or (number > 12):
                number = rand_month
            df_sub_data = df_dataset[df_dataset['WeekOfYear']==number]
            
            with st.container():
                st.write("**Zone 1 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_sub_data, x='Hour',
                                              y='Consumption_Z1',
                                              palette='Reds',
                                              x_label="Heures de la journée",
                                              title="Consommation mensuelle de la Zone 1 (KW)"))
            with st.container():
                st.write("**Zone 2 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_sub_data, x='Hour',
                                              y='Consumption_Z2',
                                              x_label="Heures de la journée",
                                              palette='Blues',
                                              title=""))
            with st.container():
                st.write("**Zone 3 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_dataset, x='Hour',
                                              x_label="Heures de la journée",
                                              y='Consumption_Z3',
                                              palette='Greens',
                                              title=""))
                
        elif window == 'Personnalisée':
            # Sélectionner un intervalle de dates
            date_range = \
                st.date_input("Choisisez une période de visualisation",
                              value=(first_day,
                                     first_day+pd.DateOffset(days=30)),
                              min_value=first_day, max_value=last_day)
            
            if len(date_range) == 1:
                date_range = (date_range[0], 
                              date_range[0]+pd.DateOffset(days=30))
            
            df_sub_data = df_dataset[date_range[0]:date_range[1]]
            
            with st.container():
                st.write("**Zone 1 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_sub_data, x='Hour',
                                              y='Consumption_Z1',
                                              palette='Reds',
                                              x_label="Heures de la journée",
                                              title="Consommation mensuelle de la Zone 1 (KW)"))
            with st.container():
                st.write("**Zone 2 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_sub_data, x='Hour',
                                              y='Consumption_Z2',
                                              x_label="Heures de la journée",
                                              palette='Blues',
                                              title=""))
            with st.container():
                st.write("**Zone 3 : consommation en fonction des heures**")
                st.pyplot(sns_display_boxplot(df_dataset, x='Hour',
                                              x_label="Heures de la journée",
                                              y='Consumption_Z3',
                                              palette='Greens',
                                              title=""))
            