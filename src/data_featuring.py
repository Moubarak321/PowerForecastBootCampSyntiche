#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:10:33 2023

@author: syntychefabien
"""

def create_datetime_features(df_data):
    """
    Permet de créer des informations supplémentaires sur les dates des
    différents enregistrements.

    Parameters
    ----------
    df_data : pandas.DataFrame
        Dataframe contenant les données.

    Returns
    -------
    df_data : pandas.DataFrame
        Dataframe contenant les colonnes d'entrée et de nouvelles avec des
        détails plus complets sur les dates.

    """
    df_data = df_data.copy()
    df_data['Hour'] = df_data.index.hour
    df_data['DayOfWeek'] = df_data.index.dayofweek
    #df_data['Quarter'] = df_data.index.quarter
    df_data['Month'] = df_data.index.month
    df_data['Year'] = df_data.index.year
    df_data['DayOfYear'] = df_data.index.dayofyear
    #df_data['DayOfMonth'] = df_data.index.day
    df_data['WeekOfYear'] = df_data.index.isocalendar().week
    
    return df_data


def add_consumption_average(df_data, rolling_hours=1):
    df_data['Z1_Mean_Consumption_{}H'.format(rolling_hours)] = \
        df_data['Consumption_Z1'].rolling('{}H'.format(rolling_hours)).mean()
    df_data['Z2_Mean_Consumption_{}H'.format(rolling_hours)] = \
        df_data['Consumption_Z2'].rolling('{}H'.format(rolling_hours)).mean()
    df_data['Z3_Mean_Consumption_{}H'.format(rolling_hours)] = \
        df_data['Consumption_Z3'].rolling('{}H'.format(rolling_hours)).mean()
    
    return df_data