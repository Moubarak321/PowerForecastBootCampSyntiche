#!/usr/bin/env python3

"""

"""

def display_prediction(df_test, target_col_name,
                       pred_col_name=["prediction"],
                       figsize=(14, 8)):
    """
    Afficher les résultats de la prédiction.

    Parameters
    ----------
    df_test : pandas.DataFrame
        Matrice des données de test: vérité terrain et données prédites.
    target_col_name : list
        Liste contenant le nom de la colonne de la donnée initiale.
    pred_col_name : list
        Liste contenant le nom de la colonne de la donnée prédite.
        La valeur par défaut est ["prediction"].
    figsize : tuple, optional
        Taille des figures pour présenter les résultats. 
        La valeur par défaut est (14, 8).

    Returns
    -------
    None.

    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from src.data_featuring import create_datetime_features 
    
    # Ajouter les données concernant l'heure et le mois à la matrice des résultats
    df_test = create_datetime_features(df_test)
    
    # Afficher les séries temporelles
    fig_ts, axis_ts = plt.subplots(1,1, figsize=figsize, sharey=True,
                             sharex=True, constrained_layout=True)  
    df_test[target_col_name].plot(figsize=figsize, label='Données initiales', ax=axis_ts)
    df_test[pred_col_name].plot(ax=axis_ts, style='.', label='Prédictions')
    axis_ts.legend()
    axis_ts.set_title('Données initials et données prédites')
    
    # Afficher les données sous forme de boîtes à moustache
    # Affichage par mois
    fig_bp_m, axis_bp_m = plt.subplots(1, 2, figsize=(14,8), sharey=True,
                             sharex=True, constrained_layout=True)    
    sns.boxplot(data=df_test, x='Month', y=target_col_name, 
                ax=axis_bp_m[0], palette='Greens')
    sns.boxplot(data=df_test, x='Month', y='prediction', 
                ax=axis_bp_m[1], palette='Reds')
    axis_bp_m[0].set_title('Données réelles de {} par mois'.format(target_col_name))
    axis_bp_m[1].set_title('Prédictions de {} par mois'.format(target_col_name))
    
    # Affichage par mois
    fig_bp_h, axis_bp_h = plt.subplots(2, 1, figsize=(14,8), sharey=True,
                             sharex=True, constrained_layout=True) 
    sns.boxplot(data=df_test, x='Hour', y=target_col_name, 
                ax=axis_bp_h[0], palette='Greens')
    sns.boxplot(data=df_test, x='Hour', y='prediction', 
                ax=axis_bp_h[1], palette='Reds')
    axis_bp_h[0].set_title('Données réelles de {} par heures'.format(target_col_name))
    axis_bp_h[1].set_title('Prédictions de {} par heures'.format(target_col_name))
