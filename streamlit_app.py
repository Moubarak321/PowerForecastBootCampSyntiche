import streamlit as st
import validators
from pages_interface.data_analysis import data_analysis
from pages_interface.predictions import predictions

st.set_page_config(
    page_title=f"Prédiction de la consommation énergétique",
    page_icon=":shark:PowerForecast", 
)


def draw_main_page():
    st.write(f"""
# PowerForecast

La prédiction de la consommation énergétique est un réel enjeu pour les gestionnaires de réseau.

Nous proposons de faire une prévision instantanée de la consommation énergétique afin d'adopter la meilleure stratégie de stockage et de distribution.
                                                                    
Les modèles de ce projet ont été construits sur les données de la ville de Tétouan au Maroc. Les données de consommation sont données en KW pour 3 zones différentes: Zone 1, Zone 2 et Zone 3.
Tétouan est une ville du Maroc avec une population de 380 000 habitants, occupant une superficie de 11 570 km². La ville est située dans la partie nord du pays et fait face à la mer Méditerranée.
Le climat est particulièrement chaud et humide pendant l'été.

Le mix énergétique du Maroc est constitué d'un mélange d'énergie fossiles et renouvelables. 
Une bonne prédiction de la consommation permet au gestionnaire de réseau d'adapter sa stratégie de stockage.
""")

    st.info("""
        :point_left: **Pour découvrir les résultats du projet, utilisez le menu latéral.**
    """)


pages = {
    "Présentation": draw_main_page,
    "Analyse exploratoire": data_analysis,
    "Prédictions": predictions,
}

# Draw sidebar

st.sidebar.title("")
selected_pages = st.sidebar.radio("", list(pages.keys()))

# Draw main page
pages[selected_pages]()
