# -*- coding: utf-8 -*-
"""
Ce fichier contient les fonctions nécessaires pour charger et pré-traiter
les données.
"""


def load_dataset(data_filename, col_sep=",", b_rename_cols=True):
    """
    Permet de charger la base de données.
    La colonne relative aux dates des différents enregistrements est
    transformée en indices pour faciliter les opérations adaptées aux séries
    temporelles.

    Parameters
    ----------
    data_filename : str
        Nom du fichier contenant les données.
    col_sep : str, optional
        Charactère de séparation des différentes colonnes dans le fichier.
        La valeur par défaut est ",".
    b_rename_cols : bool, optional
        Booléen pour indiquer si les colonnes doivent être renommées ou non
        après le chargement des données. La valeur par défaut est True.

    Returns
    -------
    df_dataset : pandas.DataFrame
        Données chargées sous forme de séries temporelles.

    """
    import pandas as pd
    
    # Load dataset
    df_dataset = pd.read_csv(data_filename, sep=col_sep)

    # Renommer les colonnes
    if b_rename_cols:
        col_names = {"DateTime": "Datetime", 
                     "Wind Speed": "WindSpeed",
                     "general diffuse flows": "GeneralDiffuseFlows", 
                     "diffuse flows": "DiffuseFlows",
                     "Zone 1 Power Consumption": "Consumption_Z1",
                     "Zone 2 Power Consumption": "Consumption_Z2",
                     "Zone 3 Power Consumption": "Consumption_Z3"}

        # Appliquer les nouvelles colonnes
        df_dataset.rename(columns=col_names, inplace=True)

   
    # Ordonner les enregistrements en se basant sur le temps
    df_dataset.set_index('Datetime').sort_index()
    
    # Transformer la donnée en série temporelle (index ==> temps)
    df_dataset = df_dataset.set_index('Datetime')
    
    # Convertir la donnée relatve au temps dans le format adapté
    df_dataset.index = pd.to_datetime(df_dataset.index)
    
    return df_dataset


