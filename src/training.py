#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

def build_sequences(df_data, df_targets, win_size_hours=2,
                    time_sampling_mins=10):
    """
    Permet de créer des séquences temporelles.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données d'entrée.
    df_targets : pandas.DataFrame
        La matrice des données à prédire.
    win_size_hours : int, optional
        La durée en heures de la séquence. La valeur par défaut est 2.
    time_sampling_mins : int, optional
        Le temps d'échantillonnage en minutes. 
        La valeur par défaut est 10.

    Returns
    -------
    numpy.array
        Les séquences d'entrée sélectionnées.
    numpy.array
        Les données de sortie correspondant aux séquences d'entée.
    targets_idx : list
        Les dates correspondant aux données de sortie.

    """
    from datetime import timedelta
    import numpy as np

    inputs = []
    targets = []
    targets_idx = []
    
    # Création d'une fénêtre glissante pour la sélection des séquences
    rol_win = df_data.rolling('{}H'.format(win_size_hours))
    valid_win_dur_mins = (win_size_hours*60) - time_sampling_mins
    valid_idx = df_data.index
    for seq in rol_win:
        seq_idx = seq.index
        win_dur_mins = ((seq_idx[-1] - seq_idx[0]).seconds) / 60
        target_idx = seq_idx[-1] + timedelta(minutes=time_sampling_mins)
        if (win_dur_mins == valid_win_dur_mins) & (target_idx in valid_idx):
            inputs.append(seq.to_numpy())
            targets.append(np.array(df_targets.loc[target_idx].to_numpy()))
            targets_idx.append(target_idx)
            
    return np.array(inputs), np.array(targets), targets_idx


def build_train_test_univariate_sequences(df_data, feat_name,
                                          win_size_hours, time_sampling_mins,
                                          train_ratio=0.8, scaler_str="Std"):

    
    assert (train_ratio>0.6) & (train_ratio<=1),\
    "Le ratio de l\'ensemble d'apprentissage doit être dans l\'intervalle ]0.6, 1]"
    
    assert (scaler_str.lower()=="minmax") | (scaler_str.lower()=="std") | \
    (scaler_str.lower()=="standard"), "Seules les normalisations 'MinMax' et 'Std' sont gérées par cette fonction"
    
    import math
    import matplotlib.pyplot as plt
    
    # Définir la base d'apprentissage en prenant un ratio de l'ensemble des données disponibles
    train_end_date = df_data.index[math.ceil(df_data.shape[0]*train_ratio)]
    df_train = df_data[[feat_name]].loc[:train_end_date]

    # La base de test est constituée du reste des données
    df_test = df_data[[feat_name]].loc[train_end_date:]
    
    # Récupérer les indices temporelles des deux ensembles
    train_index = df_train.index
    test_index = df_test.index
    
    d_train_test = dict()
    d_train_test["NonScaled"] = None
    
    if scaler_str.lower()=="minmax":
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        
        # Définir la normalisation
        scaler = MinMaxScaler()     
        
    elif scaler_str.lower()=="std" or scaler_str.lower()=="standard":
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        # Définir la normalisation
        scaler = StandardScaler()
        
    # Appliquer la normalisaton
    train_inputs = scaler.fit_transform(df_train)
    test_inputs = scaler.transform(df_test)

    # Convertir les valeurs normalisées en DataFrame
    df_train = pd.DataFrame(data=train_inputs, index=train_index,
                            columns=[feat_name])
    df_test = pd.DataFrame(data=test_inputs, index=test_index,
                           columns=[feat_name])       
      
    # Création des séquences temporelles
    X_train, y_train, idx_train = \
        build_sequences(df_data=df_train,
                        df_targets=df_train,
                        win_size_hours=win_size_hours,
                        time_sampling_mins=time_sampling_mins)

    X_test, y_test, idx_test = \
        build_sequences(df_data=df_test,
                        df_targets=df_test,
                        win_size_hours=win_size_hours,
                        time_sampling_mins=time_sampling_mins)

    d_train_test["Scaled"] = {"train":{"Inputs":X_train,
                                       "Target":y_train,
                                       "Datetimes": idx_train},
                                 "test":{"Inputs":X_test,
                                         "Target":y_test,
                                         "Datetimes": idx_test}}
        
    
    # Affichage du découpage de la base de données
    fig_ts, axis_ts = plt.subplots(1,1, figsize=(14, 8), sharey=True,
                             sharex=True, constrained_layout=True)  
    df_train.plot(ax=axis_ts)
    df_test.plot(ax=axis_ts)
    axis_ts.legend(["Données d'appreentissage", 'Données de test'])
    axis_ts.set_title("Découpage temporel données de test et d'apprentissage")

    return X_train, y_train, idx_train, X_test, y_test, idx_test, scaler


def define_LSTM_net(seq_length, n_features, n_targets,
                    layers=[100, 50, 10], fn_act="relu"):

    assert isinstance(layers, list), "La taille des couches cachées doit être une liste."
    assert len(layers) >= 2, "Il faut au moins deux couches dans le réseau"
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
        
    model_lstm = Sequential()
    model_lstm.add(LSTM(layers[0], activation=fn_act,
                   input_shape=(seq_length, n_features),
                   return_sequences=True))
    
    for idx in range(1, len(layers)-1):
        model_lstm.add(LSTM(layers[idx], activation=fn_act, return_sequences=True))
         
    model_lstm.add(LSTM(layers[-1], activation=fn_act))
    model_lstm.add(Dense(n_targets))

    # Affichage du réseau de neurones
    model_lstm.summary()
    
    return model_lstm


def train_lstm_univariate(X_train, y_train, X_test, y_test,
                          d_learning_params, results_dir):
    import os
    from datetime import datetime
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    from src.training import define_LSTM_net
    from tensorflow.keras import callbacks
    import numpy as np
    
    # Récupérer la date et l'heure pour la création du fichier du modèle
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, current_datetime)
    params_file = os.path.join(results_dir, 
                               "lstm_params_{}.pkl".format(str(current_datetime)))

    # Création du répertoire de sauvegarde des résultats s'il n'existe pas déjà
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(params_file, 'wb') as fp:
        pickle.dump(d_learning_params, fp)

    model_file = os.path.join(results_dir, "models", "lstm_{}.h5".format(current_datetime))
    log_dir = os.path.join(results_dir, "logs")

    # Définir le comportement pendant l'apprentissage
    # Utilisation de tensorboard pour suivre les performance de l'apprentissage non indispensable
    my_callbacks = [callbacks.ModelCheckpoint(filepath=model_file,
                                              monitor='val_loss', mode='auto',
                                              save_best_only=True, verbose=0,
                                              save_weights_only=False),
                    callbacks.EarlyStopping(monitor="val_loss", mode='auto',
                                           patience=15, verbose=0),
                    callbacks.TensorBoard(log_dir=log_dir)]
    
    # Création du réseau de neurones
    model = define_LSTM_net(seq_length=X_train.shape[1],
                            n_features=X_train.shape[2],
                            n_targets=y_train.shape[1],
                            layers=d_learning_params['layers'],
                            fn_act=d_learning_params['activation_fn'])

    # compilation du modèle avec les différentes fonctions d'optimisation, de coût et de précision
    model.compile(optimizer=d_learning_params['optimizer'], 
                  loss=d_learning_params['loss_fn'],
                  metrics=d_learning_params['metrics'])

    # Entrainement du modèle
    history = model.fit(X_train, y_train,
                        batch_size = d_learning_params['batch_size'],
                        epochs=d_learning_params['epochs'],
                        validation_split=d_learning_params['val_ratio'],
                        callbacks=my_callbacks, verbose=1)

    # Affichage de la loss et de la métrique
    fig_perf, axis_perf = plt.subplots(2,1, figsize=(14, 8),
                                      sharex=True, constrained_layout=True)

    axis_perf[0].plot(history.history['loss'], label='Training loss')
    axis_perf[0].plot(history.history['val_loss'], label='Validation loss')
    axis_perf[0].legend()
    axis_perf[0].set_title("Evolution de la fonction de coût pendant l'apprentissage")
    
    axis_perf[1].plot(history.history[d_learning_params['metrics'][0]],
                      label='Training {}'.format(d_learning_params['metrics'][0]))
    axis_perf[1].plot(history.history['val_{}'.format(d_learning_params['metrics'][0])],
                      label='Validation {}'.format(d_learning_params['metrics'][0]))
    axis_perf[1].legend()
    axis_perf[0].set_title("Evolution de la métrique'{}' pendant l'apprentissage".format(d_learning_params['metrics'][0]))

    
    # Afficher les performances du modèle
    y_test_pred = d_learning_params['scaler'].inverse_transform(model.predict(X_test))
    y_train_pred = d_learning_params['scaler'].inverse_transform(model.predict(X_train))
    
    y_test_rescaled = d_learning_params['scaler'].inverse_transform(y_test)
    y_train_rescaled = d_learning_params['scaler'].inverse_transform(y_train)
    
    # Inverse scaling
    
    scores = dict()
    scores['R2'] = {"train": np.round(r2_score(y_train_rescaled, y_train_pred), 3),
                    "test": np.round(r2_score(y_test_rescaled, y_test_pred), 3)}
    scores['MAE'] = {"train": np.round(mean_absolute_error(y_train_rescaled,
                                                           y_train_pred), 3),
                     "test": np.round(mean_absolute_error(y_test_rescaled,
                                                          y_test_pred), 3)}
    
    return model, scores
    

def train_lstm(X_train, y_train, X_test, y_test, d_learning_params,
               results_dir):
    import os
    from datetime import datetime
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    from src.training import define_LSTM_net
    from tensorflow.keras import callbacks
    import numpy as np
    
    # Récupérer la date et l'heure pour la création du fichier du modèle
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, current_datetime)
    params_file = os.path.join(results_dir, 
                               "lstm_params_{}.pkl".format(str(current_datetime)))

    # Création du répertoire de sauvegarde des résultats s'il n'existe pas déjà
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(params_file, 'wb') as fp:
        pickle.dump(d_learning_params, fp)

    model_file = os.path.join(results_dir, "models", "lstm_{}.h5".format(current_datetime))
    log_dir = os.path.join(results_dir, "logs")

    # Définir le comportement pendant l'apprentissage
    # Utilisation de tensorboard pour suivre les performance de l'apprentissage non indispensable
    my_callbacks = [callbacks.ModelCheckpoint(filepath=model_file,
                                              monitor='val_loss', mode='auto',
                                              save_best_only=True, verbose=0,
                                              save_weights_only=False),
                    callbacks.EarlyStopping(monitor="val_loss", mode='auto',
                                           patience=15, verbose=0),
                    callbacks.TensorBoard(log_dir=log_dir)]
    
    # Création du réseau de neurones
    model = define_LSTM_net(seq_length=X_train.shape[1],
                            n_features=X_train.shape[2],
                            n_targets=y_train.shape[1],
                            layers=d_learning_params['layers'],
                            fn_act=d_learning_params['activation_fn'])

    # compilation du modèle avec les différentes fonctions d'optimisation, de coût et de précision
    model.compile(optimizer=d_learning_params['optimizer'], 
                  loss=d_learning_params['loss_fn'],
                  metrics=d_learning_params['metrics'])

    # Entrainement du modèle
    history = model.fit(X_train, y_train,
                        batch_size = d_learning_params['batch_size'],
                        epochs=d_learning_params['epochs'],
                        validation_split=d_learning_params['val_ratio'],
                        callbacks=my_callbacks, verbose=1)

    # Affichage de la loss et de la métrique
    fig_perf, axis_perf = plt.subplots(2,1, figsize=(14, 8),
                                      sharex=True, constrained_layout=True)

    axis_perf[0].plot(history.history['loss'], label='Training loss')
    axis_perf[0].plot(history.history['val_loss'], label='Validation loss')
    axis_perf[0].legend()
    axis_perf[0].set_title("Evolution de la fonction de coût pendant l'apprentissage")
    
    axis_perf[1].plot(history.history[d_learning_params['metrics'][0]],
                      label='Training {}'.format(d_learning_params['metrics'][0]))
    axis_perf[1].plot(history.history['val_{}'.format(d_learning_params['metrics'][0])],
                      label='Validation {}'.format(d_learning_params['metrics'][0]))
    axis_perf[1].legend()
    axis_perf[0].set_title("Evolution de la métrique'{}' pendant l'apprentissage".format(d_learning_params['metrics'][0]))

    
    # Afficher les performances du modèle
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = dict()
    scores['R2'] = {"train": np.round(r2_score(y_train, y_train_pred), 3),
                    "test": np.round(r2_score(y_test, y_test_pred), 3)}
    scores['MAE'] = {"train": np.round(mean_absolute_error(y_train,
                                                           y_train_pred), 3),
                     "test": np.round(mean_absolute_error(y_test,
                                                          y_test_pred), 3)}
    
    return model, scores
    


def build_train_test_datasets(df_data,
                              input_cols=['Temperature', 'Humidity', 
                                          'WindSpeed', 'hour'], 
                              target_cols=["Consumption_Z1"],
                              train_ratio=0.8, scaler_str="Std"):
    """
    Permet de créer les ensembles d'apprentissage et de test en appliquant une
    normalisation si précisé en paramètres.

    Parameters
    ----------
    df_data : pandas.DataFrame
        La matrice des données d'entrée passée en paramètre.
    input_cols : list, optional
        La liste des colonnes considérées comme les entrées du modèle.
        La valeur par défaut est he default ['Temperature', 'Humidity', 
                                             'WindSpeed', 'hour'].
    target_col : str, optional
        Colonne considérée comme étant la valeur à prédire. 
        La valeur par défaut est "Consumption_Z1".
    train_ratio : float, optional
        La proportion des données devant être utilisé pour l'apprentissage.
        La valeur par défaut est 0.8.
    scaler_str : str, optional
        Un chaine de caractères indiquant quelle normalisation devrait être 
        appliquée aux données. Si la valur est valide, la normalisation est
        calculée sur les données d'apprentissage et appliquées sur l'ensemble
        des données. La valeur par défaut est 'Std' pour appliquer une
        normalisation centrée réduite.

    Returns
    -------
    d_train_test : dict
        Dictionnaire contenant les données d'apprentissage et de test
        normalisées et non normalisées.
    scaler : sklearn.preprocessing.MinMaxScaler
        Les coefficients de normalisation de l'opération "Min-Max" appliquée
        aux données d'apprentissage.

    """
    
    assert (train_ratio>0.6) & (train_ratio<=1),\
    "Le ratio de l\'ensemble d'apprentissage doit être dans l\'intervalle ]0.6, 1]"
    
    import math
    import matplotlib.pyplot as plt
    
    # Définir la base d'apprentissage en prenant un ratio de l'ensemble des données disponibles
    train_end_date = df_data.index[math.ceil(df_data.shape[0]*train_ratio)]
    train_inputs = df_data[input_cols].loc[:train_end_date]
    train_targets = df_data[target_cols].loc[:train_end_date]

    # La base de test est constituée du reste des données
    test_inputs = df_data[input_cols].loc[train_end_date:]
    test_targets = df_data[target_cols].loc[train_end_date:]
    
    d_train_test = dict()
    d_train_test["NonScaled"] = {"train":{"Inputs":train_inputs,
                                          "Target":train_targets},
                                 "test":{"Inputs":test_inputs,
                                         "Target":test_targets}}
    
    if scaler_str is None:
        scaler = None
        d_train_test["Scaled"] = None
    
    elif scaler_str.lower()=="minmax":
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd

        # Récupérer les indices temporelles des deux ensembles
        train_index = train_inputs.index
        test_index = test_inputs.index
        
        # Appliquer la normalisaton
        scaler = MinMaxScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

        # Convertir les valeurs normalisées en DataFrame
        train_inputs = pd.DataFrame(data=train_inputs, index=train_index,
                                    columns=input_cols)
        test_inputs = pd.DataFrame(data=test_inputs, index=test_index,
                                   columns=input_cols)
                                    
        d_train_test["Scaled"] = {"train":{"Inputs":train_inputs,
                                           "Target":train_targets},
                                 "test":{"Inputs":test_inputs,
                                         "Target":test_targets}}
        
    elif scaler_str.lower()=="std" or scaler_str.lower()=="standard":
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        # Récupérer les indices temporelles des deux ensembles
        train_index = train_inputs.index
        test_index = test_inputs.index
        
        # Appliquer la normalisaton
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

        # Convertir les valeurs normalisées en DataFrame
        train_inputs = pd.DataFrame(data=train_inputs, index=train_index,
                                    columns=input_cols)
        test_inputs = pd.DataFrame(data=test_inputs, index=test_index,
                                   columns=input_cols)
                                    
        d_train_test["Scaled"] = {"train":{"Inputs":train_inputs, "Target":train_targets},
                                 "test":{"Inputs":test_inputs, "Target":test_targets}}
    else:
        scaler = None
        d_train_test["Scaled"] = None
        
    
    # Affichage du découpage de la base de données
    fig_ts, axis_ts = plt.subplots(1,1, figsize=(14, 8), sharey=True,
                             sharex=True, constrained_layout=True)  
    train_targets.plot(ax=axis_ts)
    test_targets.plot(ax=axis_ts)
    axis_ts.legend(["Données d'appreentissage", 'Données de test'])
    axis_ts.set_title("Découpage temporel données de test et d'apprentissage")

    return d_train_test, scaler


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=500,
                  b_feat_importance=True, b_verbose=True): 
    
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    # Définir le modèle
    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                             n_estimators=n_estimators,
                             objective='reg:squarederror',
                             max_depth=10,
                             learning_rate=0.01, 
                             random_state=48)
    
    # Entraîner le modèle sur l'ensemble de test et l'évaluer également sur
    # l'ensemble de test
    if b_verbose:
        verbose = 100
    else:
        verbose = 0
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=verbose)
    
    # Afficher les performances du modèle
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = dict()
    scores['R2'] = {"train": np.round(model.score(X_train, y_train), 3),
                    "test": np.round(model.score(X_test, y_test), 3)}
    scores['MAE'] = {"train": np.round(mean_absolute_error(y_train,
                                                           y_train_pred), 3),
                     "test": np.round(mean_absolute_error(y_test,
                                                          y_test_pred), 3)}

    print("Performance du modèle")
    print("\t --> Score R2 (Le modèle parfait a un score de 1)")
    print("\t \t --> Ensemble d'apprentissage : {}".format(scores['R2']['train']))
    print("\t \t --> Ensemble de test : {}".format(scores['R2']['test']))
    print("\n")
    print("\t --> MAE (Erreur moyenne absolue) en KW")
    print("\t \t --> Ensemble d'apprentissage : {}".format(scores['MAE']['train']))
    print("\t \t --> Ensemble de test : {}".format(scores['MAE']['test']))
    
    if b_feat_importance:
        import pandas as pd
        # Calculer l'importance des données d'entrée dans la prédiction
        feat_importances = pd.DataFrame(data=model.feature_importances_,
                                        index=X_train.columns,
                                        columns=['Feat_importance'])
        # Trier les poids des variables d'entrée afn d'avoir un affichage
        # explicite
        feat_importances = feat_importances.sort_values('Feat_importance')
        # Afficher le poids des différntes données d'entrée
        feat_importances.plot(kind='barh', title='Feature Importance')
        
    return model, scores
        

        
def apply_cross_validation(model, X_train, y_train, n_splits=10, n_repeats=3):
    
    assert n_splits >=2, "n_splits doit être supérieur ou égal à 2."
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import RepeatedKFold
    
    if model is None:
        import xgboost as xgb
        model = xgb.XGBRegressor(booster='gbtree', 
                                 objective='reg:squarederror')
    
    # define model evaluation method
    cross_val = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                              random_state=1)
    # evaluate model
    cv_results =\
        cross_validate(model, X_train, y_train, 
                       return_estimator=True, return_train_score=True,
                       scoring=('r2', 'neg_mean_absolute_error'), 
                       cv=cross_val, n_jobs=-1)
    
    return cv_results

        