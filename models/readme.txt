Les modèles validés se trouvent dans les dossiers Zone1, Zone2 et Zone3. Les autres dossiers au nom d'un modèle correspondent à ceux obtenus pendant l'entrainement avec les notebooks. Ils contiennent également les prédictions et les paramètres d'entrainement. 

Attention les modèles LSTM sont des fichiers .h5 à charger avec la fonction `load` de tensorflow.
Dans les dossiers j'ai mis également un ".pkl" contenant les paramètres du modèle et le scaler notamment utile pour la version "univariate". Le scale est également fait sur les données de sortie ;). Il faut bien qu'ils apprennent à utiliser la fonction "inverse" de scale.


Pour le modèle XGBoost c'est un dictionnaire qui est sauvegardé avec le scaler. En l'occurence il n'y en a pas. Le meilleur modèle de la validation est celui qui a été sauvegardé. Pour le récupérer, il suffit de passer par la clé "cross_val_best_model".