#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce fichier regroupe quelques fonctions utiles pour analyser les données.
"""

def apply_sanity_check(df_dataset):
    import pandas as pd
    import numpy as np

    assert isinstance(df_dataset.index, pd.DatetimeIndex), \
    "Les indices de la matrice de données doivent être le temps"

    d_sanity_check = dict()
    # Compter les valeurs invalides de la base de données
    d_sanity_check["Valeurs_invalides"] = dict(df_dataset.isnull().sum())

    # Vérifier la fréquence d'échantillonnage
    time_diff = df_dataset.index.to_series().diff().astype('timedelta64[s]').values

    # Supprimer la première valeur parce que la différence pour le premier pas de temps
    time_diff = np.sum(np.diff(time_diff[1:])).astype(float)
       
    if time_diff < 10e-3:
        d_sanity_check['Periode_echantillonnage'] = "Uniforme"
    else:
        d_sanity_check['Periode_echantillonnage'] = "Non Uniform"
  
    return d_sanity_check


def plot_corr(df, figsize=(10, 10)):
    """
    Permet d'afficher la matrice de corrélation de la matrice des données
    passée en paramètres.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrice de données d'entrée.
    figsize : tuple, optional
        Dimensions de la figure utilisée pour l'affichage de la matrice de
        corrélation. La valeur par défaut est (10, 10).

    Returns
    -------
    None.

    """
    # Analyser la matrice de corrélation
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Calcul de la matrice de corrélation
    corr = df.corr()

    # Créer une figure pour la matrice de corrélation
    fig, ax = plt.subplots(figsize=figsize)

    # Générer une palette de couleur personnalisée
    cmap = sns.diverging_palette(210, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0,square=True,
                linewidths=.5, cbar_kws={"shrink": .5})
    
    
def display_consumption_monthly(df_data, figsize=(20, 15), month_num="all"):
    """
    Permet d'afficher les séries temporelles des consommations mensuelles
    des 3 zones.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données passée en paramètres.
    figsize : tuple, optional
        La taille de la fenêtre utilisée pour l'affichage. La valeur par 
        défaut est (20, 15).
    month_num : int, optional
        Le numéro de mois concerné par l'affichage. Cette valeur doit être
        posiive et inférieure ou égale à 12. Pour afficher toutes
        les données utilisées on peut indiquer "all". 
        La valeur par défaut est "all".

    Raises
    ------
    ValueError
        Exception levée quand la valeur du numéro du mois n'est pas valide.

    Returns
    -------
    None.

    """
    
    assert "Month" in df_data.columns, \
        "La colonne \'Month' n\'est pas présente dans la matrice de données"
    
    if isinstance(month_num, str):
        if month_num.lower()=="all":
            df_sub_data = df_data.copy()
        else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    elif isinstance(month_num, int):
        if (month_num <= 12) & (month_num > 0):
            df_sub_data = df_data[df_data["Month"]==month_num]
        else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    else:
            df_sub_data = None
            print("Le numéro du mois de l'année n'est pas valide : "
                  "valeur positive et inférieure à 12.")
            raise ValueError
    
    if df_sub_data is not None:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(3, 1, figsize=figsize,
                                 sharex=True, sharey=True,
                                 constrained_layout=True)
        df_sub_data['Consumption_Z1'].plot(ax=axis[0],
                                           title='Zone 1 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
        df_sub_data['Consumption_Z2'].plot(ax=axis[1],
                                           title='Zone 2 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
        df_sub_data['Consumption_Z3'].plot(ax=axis[2],
                                           title='Zone 3 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
 
    
def display_consumption_daily(df_data, figsize=(20, 15), day_num="all"):
    """
    Permet d'afficher les séries temporelles relatives au consommations
    journalières.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La base de données passée en paramètres.
    figsize : tuple, optional
        Dimensions de la figure servant à l'affichage des consommations
        journalières. La valeur par défaut est (20, 15).
    day_num : int, optional
        Numéro du jour de l'année concerné par l'affichage. Cette valeur doit
        être positive et inférieure ou égale à 365. 
        La valeur par défaut est "all".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    assert "DayOfYear" in df_data.columns,\
        "La colonne \'DayOfYear' n\'est pas présente dans la matrice de données"
    
    if isinstance(day_num, str):
        if day_num.lower()=="all":
            df_sub_data = df_data.copy()
        else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    elif isinstance(day_num, int):
        if (day_num <= 365) & (day_num > 0):
            df_sub_data = df_data[df_data["DayOfYear"]==day_num]
        else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    else:
            df_sub_data = None
            print("Le numéro du jour de l'année n'est pas valide : "
                  "valeur positive et inférieure à 365.")
            raise ValueError
    
    if df_sub_data is not None:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(1, 1, figsize=figsize,
                                 sharex=True, sharey=True,
                                 constrained_layout=True)
        df_sub_data['Consumption_Z1'].plot(ax=axis,
                                           title='Zone 1 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
        df_sub_data['Consumption_Z2'].plot(ax=axis,
                                           title='Zone 2 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
        df_sub_data['Consumption_Z3'].plot(ax=axis,
                                           title='Zone 3 power consumption (KW)',
                                           ylabel="Power consumption (KW)")
        
        
def display_consumption_resume(df_data, scale="Hours", figsize=(10, 8)):
    """
    Permet d'afficher les consommations de façon compacte à l'aide de boite à
    moustaches sur une echelle mensuelle ou journalière.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice contenant les données d'entrée passées en paramètres.
    scale : str, optional
        L'achelle de temps sur laquelle on veut analyser les enregistrements.
        Seules les échelles journalière (Hours) et mensuelle (Months) 
        sont disponibles. La valeur par défaut est "Hours".
    figsize : tuple, optional
        Dimensions de la figure utilisée pour l'affichage. La valeur par 
        défaut est (10, 8).

    Raises
    ------
    ValueError
        DESCRIPTION.
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if isinstance(scale, str):
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if scale.lower()=="hours":
            fig, axis = plt.subplots(3,1, figsize=figsize, sharey=True,
                                     sharex=True, constrained_layout=True)    
        
            sns.boxplot(data=df_data, x='Hour', y='Consumption_Z1', 
                        ax=axis[0], palette='Reds')
            sns.boxplot(data=df_data, x='Hour', y='Consumption_Z2', 
                        ax=axis[1], palette='Blues')
            sns.boxplot(data=df_data, x='Hour', y='Consumption_Z3', 
                        ax=axis[2], palette='Greens')
            axis[0].set_title('Zone 1 consumption power (KW) by Hour')
            axis[1].set_title('Zone 2 consumption power (KW) by Hour')
            axis[2].set_title('Zone 3 consumption power (KW) by Hour')
            
            for i in range(3):
                axis[i].set_xlabel('Hours')
                axis[i].set_ylabel('Consumption power (KW)')
        elif scale.lower()=="months":
            fig, axis = plt.subplots(3,1, figsize=figsize, sharey=True,
                                     sharex=True, constrained_layout=True) 
            sns.boxplot(data=df_data, x='Month', y='Consumption_Z1', 
                        ax=axis[0], palette='Reds')
            sns.boxplot(data=df_data, x='Month', y='Consumption_Z2',
                        ax=axis[1], palette='Blues')
            sns.boxplot(data=df_data, x='Month', y='Consumption_Z3', 
                        ax=axis[2], palette='Greens')
            axis[0].set_title('Zone 1 consumption power (KW) by Hour')
            axis[1].set_title('Zone 2 consumption power (KW) by Hour')
            axis[2].set_title('Zone 3 consumption power (KW) by Hour')
            
            for i in range(3):
                axis[i].set_xlabel('Months')
                axis[i].set_ylabel('Consumption power (KW)')
        else:
            print("La valeur de l'échelle de temps n'est pas conforme: '"
                  "Hours' ou 'Months'.")
            raise ValueError
    else:
        print("L\'échelle de temps doit être une chaine de caractères.")
        raise TypeError
              

