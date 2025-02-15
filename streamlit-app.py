import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from model_lstm import predict

# cd "C:\Python\Datascientest\Projet\8-Streamlit"
# streamlit run .\streamlit-app.py

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 40%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    "# Modélisation et prédiction des accidents routiers"
    st.image("img/logo3.jpeg")
    

def banner():

    #image = [
    #    "img/logo-datascientest.png",
    #    "img/logo-minesparis.png",
    #    "img/logo-gendarme.png"
    #]

    #col1, col2, col3 = st.columns(3)
    #col1.image(image[0], width=120)
    #col2.image(image[1], width=120)
    #col3.image(image[2], width=120)
    st.image("img/Image1.png")

def page_introduction():

    banner()
    #st.image("img/Image1.png")
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Introduction</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Contexte métier", "Méthodologie", "Le planning du projet et les jalons"]
    )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Contexte métier</h4>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Méthodologie</h4>",
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Le planning du projet et les jalons</h4>",
            unsafe_allow_html=True,
        )


def page_EAD():

    st.image("img\Image1.png")
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Analyse exploiratoire des données</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Modèle de données", "Répartition des classes", "Géolocalisation"]
    )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Modèle de données</h4>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Répartition des classes</h4>",
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Géolocalisation</h4>",
            unsafe_allow_html=True,
        )



def page_conclusion():

    st.image("img\Image1.png")
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Conclusion</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(
        ["Synthèse des résultats obtenus", "Perspectives"]
    )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Synthèse des résultats obtenus</h4>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Perspectives</h4>",
            unsafe_allow_html=True,
        )

def tab_demo_ML():
    
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Demonstration </h2>",
        unsafe_allow_html=True,
    )

    st.write(
        """ 
            In this part, we let you choose the parameters of an accident.
            \n \n
            In return, you'll see the probability evaluated by our XGBoost model for this accident to be severe. 
            You'll also see which one of the three actors in our case study would consider this accident interesting for 
            any political measure.  
            \n \n 
            For a better readability, we've only showed the five parameters with the most influence on the result 
            but you can choose every parameter by expanding below.   
            
            
            """
    )

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Model parameters </h4>",
        unsafe_allow_html=True,
    )

    # Conteneur des variables non visibles
    col58, col59, col60 = st.columns([1, 1, 1])

    with col58:
        population = st.selectbox(
            "City Size",
            [
                "village",
                "small town",
                "average city",
                "big city",
                "metropolis",
                "other",
            ],
        )
        categorie_route = st.selectbox(
            "Road Type",
            [
                "other",
                "national",
                "departmental",
                "communal",
            ],
        )

    with col59:
        conditions_atmospheriques = st.selectbox(
            "Weather conditions",
            [
                "light rain",
                "heavy rain",
                "smog",
                "dazzling",
                "cloudy",
                "other",
            ],
        )
        luminosité = st.selectbox(
            "brightness",
            [
                "dawn ou dusk",
                "night w/o public lighting",
                "night with public lighting",
                "other",
            ],
        )
    with col60:
        nb_vehicule = st.slider("number of vehicles", 0, 10, 2)

    with st.expander("other parameters", expanded=False):
        col61, col62, col63 = st.columns([1, 1, 1])

        with col61:
            age_moyen_conducteurs = st.slider("average age of drivers", 18, 100, 30)
            nb_personnes_cote_choc = st.slider(
                "number of person on the impact side", 0, 10, 1
            )

            regime_circulation = st.selectbox(
                "traffic type",
                [
                    "other",
                    "bidirectional",
                    "with separate ways",
                    "with specials ways",
                ],
            )
            type_collision = st.selectbox(
                "collision type",
                [
                    "other",
                    "3 vehicles et + - multiples",
                    "2 vehicles - back impact",
                    "2 vehicles - side impact",
                    "3 vehicles et + - chain impact ",
                    "w/o collision",
                ],
            )
            infra = st.selectbox(
                "road infrastructure",
                [
                    "other",
                    "interchange ramps",
                    "traffic circle",
                    "toll area",
                ],
            )

        with col62:
            intersection = st.selectbox(
                "intersection type",
                [
                    "other",
                    "railroad crossing",
                    "traffic roundabout ",
                    "crossroads",
                    "outside intersections",
                ],
            )

            mois = st.selectbox(
                "month",
                [
                    "march",
                    "january",
                    "february",
                    "july",
                    "august",
                    "october",
                    "december",
                    "other",
                ],
            )

            profil = st.selectbox("road profile", ["slope", "hill top", "other"])
            situation = st.selectbox(
                "accident situation",
                [
                    "other",
                    "on emergency lane",
                    "on shoulder",
                    "on sidewalk",
                    "on special lane",
                ],
            )
            surface = st.selectbox(
                "road surface",
                ["other", "with oil / greasy substance ", "wet"],
            )
        with col63:
            pres_2roues = st.checkbox("presence of a two wheeler")
            pres_EPD = st.checkbox("presence of a personal transporter")
            pres_PL = st.checkbox("presence of a HGV")
            pres_train = st.checkbox("presence of a train")
            pres_pieton = st.checkbox("presence of a pedestrian")
            abs_obstacle = st.checkbox("no obstacles")
            loc_pieton = st.checkbox(
                "pedestrian on pedestrian crossing w/o light signaling"
            )
            nuit = st.checkbox("night accident")
            route_rectiligne = st.checkbox("straight road")
            pres_homme_volant = st.checkbox("presence of a male driver")
            pres_femme_volant = st.checkbox("presence of a female driver")
            trajet_promenade = st.checkbox("promenade")
            pres_piste_cyclabe = st.checkbox("presence of a bike path")
        # ... Fin du conteneur

    result = st.button("Predict")
    if result:
        X_test = pd.DataFrame(
            columns=[
                "choc_cote",
                "ageMeanConductors",
                "nbVeh",
                "prof_2.0",
                "prof_3.0",
                "planGrp_1.0",
                "surf_2.0",
                "surf_8.0",
                "atm_2.0",
                "atm_3.0",
                "atm_5.0",
                "atm_7.0",
                "atm_8.0",
                "vospGrp_1.0",
                "catv_EPD_exist_1",
                "catv_PL_exist_1",
                "trajet_coursesPromenade_conductor_1",
                "sexe_male_conductor_1",
                "sexe_female_conductor_1",
                "intGrp_Croisement circulaire",
                "intGrp_Croisement de deux routes",
                "intGrp_Hors intersection",
                "intGrp_Passage à niveau",
                "catv_train_exist_1",
                "infra_3.0",
                "infra_5.0",
                "infra_7.0",
                "infra_9.0",
                "catr_2.0",
                "catr_3.0",
                "catr_4.0",
                "catr_9.0",
                "hourGrp_nuit",
                "lum_2.0",
                "lum_3.0",
                "lum_5.0",
                "circ_2.0",
                "circ_3.0",
                "circ_4.0",
                "nbvGrp_1",
                "nbvGrp_2",
                "nbvGrp_3",
                "nbvGrp_4+",
                "catv_2_roues_exist_1",
                "col_2.0",
                "col_3.0",
                "col_4.0",
                "col_5.0",
                "col_6.0",
                "col_7.0",
                "obsGrp_Pas d'Obstacle",
                "situ_2.0",
                "situ_3.0",
                "situ_4.0",
                "situ_6.0",
                "situ_8.0",
                "populationGrp_Grande Ville",
                "populationGrp_Métropole",
                "populationGrp_Petite Ville",
                "populationGrp_Village",
                "populationGrp_Ville Moyenne",
                "mois_label_aug",
                "mois_label_dec",
                "mois_label_fev",
                "mois_label_jan",
                "mois_label_jul",
                "mois_label_mar",
                "mois_label_oct",
                "etatpGrp_pieton_alone_1",
                "locpGrp_pieton_3_1",
            ],
            dtype="int",
        )

        X_test.loc[0, "choc_cote"] = nb_personnes_cote_choc
        X_test.loc[0, "ageMeanConductors"] = age_moyen_conducteurs
        X_test.loc[0, "nbVeh"] = nb_vehicule

        X_test.loc[0, "prof_2.0"] = 1 if profil == "slope" else 0
        X_test.loc[0, "prof_3.0"] = 1 if profil == "hill top" else 0

        X_test.loc[0, "planGrp_1.0"] = 1 if route_rectiligne else 0

        X_test.loc[0, "surf_2.0"] = 1 if surface == "wet" else 0
        X_test.loc[0, "surf_8.0"] = 1 if surface == "with oil / greasy substance" else 0

        X_test.loc[0, "atm_2.0"] = 1 if conditions_atmospheriques == "light rain" else 0
        X_test.loc[0, "atm_3.0"] = 1 if conditions_atmospheriques == "heavy rain" else 0
        X_test.loc[0, "atm_5.0"] = 1 if conditions_atmospheriques == "smog" else 0
        X_test.loc[0, "atm_7.0"] = 1 if conditions_atmospheriques == "dazzling" else 0
        X_test.loc[0, "atm_8.0"] = 1 if conditions_atmospheriques == "cloudy" else 0

        X_test.loc[0, "vospGrp_1.0"] = 1 if pres_piste_cyclabe else 0

        X_test.loc[0, "catv_EPD_exist_1"] = 1 if pres_EPD else 0
        X_test.loc[0, "catv_PL_exist_1"] = 1 if pres_PL else 0

        X_test.loc[0, "trajet_coursesPromenade_conductor_1"] = (
            1 if trajet_promenade else 0
        )

        X_test.loc[0, "sexe_male_conductor_1"] = 1 if pres_homme_volant else 0
        X_test.loc[0, "sexe_female_conductor_1"] = 1 if pres_femme_volant else 0

        X_test.loc[0, "intGrp_Croisement circulaire"] = (
            1 if intersection == "traffic roundabout" else 0
        )
        X_test.loc[0, "intGrp_Croisement de deux routes"] = (
            1 if intersection == "crossroads" else 0
        )
        X_test.loc[0, "intGrp_Hors intersection"] = (
            1 if intersection == "outside intersections" else 0
        )
        X_test.loc[0, "intGrp_Passage à niveau"] = (
            1 if intersection == "railroad crossing" else 0
        )

        X_test.loc[0, "catv_train_exist_1"] = 1 if pres_train else 0

        X_test.loc[0, "infra_3.0"] = 1 if infra == "interchange ramps" else 0
        X_test.loc[0, "infra_5.0"] = 1 if infra == "traffic circle" else 0
        X_test.loc[0, "infra_7.0"] = 1 if infra == "toll area" else 0
        X_test.loc[0, "infra_9.0"] = 1 if infra == "other" else 0

        X_test.loc[0, "catr_2.0"] = 1 if categorie_route == "national" else 0
        X_test.loc[0, "catr_3.0"] = 1 if categorie_route == "departmental" else 0
        X_test.loc[0, "catr_4.0"] = 1 if categorie_route == "communal" else 0
        X_test.loc[0, "catr_9.0"] = 1 if categorie_route == "other" else 0

        X_test.loc[0, "hourGrp_nuit"] = 1 if nuit else 0

        X_test.loc[0, "lum_2.0"] = 1 if luminosité == "dawn ou dusk" else 0
        X_test.loc[0, "lum_3.0"] = 1 if luminosité == "night w/o public lighting" else 0
        X_test.loc[0, "lum_5.0"] = (
            1 if luminosité == "night with public lighting" else 0
        )

        X_test.loc[0, "circ_2.0"] = 1 if regime_circulation == "bidirectional" else 0
        X_test.loc[0, "circ_3.0"] = (
            1 if regime_circulation == "with separate ways" else 0
        )
        X_test.loc[0, "circ_4.0"] = (
            1 if regime_circulation == "with specials ways" else 0
        )

        X_test.loc[0, "nbvGrp_1"] = 1 if nb_vehicule == 1 else 0
        X_test.loc[0, "nbvGrp_2"] = 1 if nb_vehicule == 2 else 0
        X_test.loc[0, "nbvGrp_3"] = 1 if nb_vehicule == 3 else 0
        X_test.loc[0, "nbvGrp_4+"] = 1 if nb_vehicule >= 4 else 0

        X_test.loc[0, "catv_2_roues_exist_1"] = 1 if pres_2roues else 0

        X_test.loc[0, "col_2.0"] = (
            1 if type_collision == "2 vehicles - back impact" else 0
        )
        X_test.loc[0, "col_3.0"] = (
            1 if type_collision == "2 vehicles - side impact" else 0
        )
        X_test.loc[0, "col_4.0"] = (
            1 if type_collision == "3 vehicles et + - chain impact " else 0
        )
        X_test.loc[0, "col_5.0"] = (
            1 if type_collision == "3 vehicles et + - multiples" else 0
        )
        X_test.loc[0, "col_6.0"] = 1 if type_collision == "w/o collision" else 0
        X_test.loc[0, "col_7.0"] = 1 if type_collision == "other" else 0

        X_test.loc[0, "obsGrp_Pas d'Obstacle"] = 1 if abs_obstacle else 0

        X_test.loc[0, "situ_2.0"] = 1 if situation == "on emergency lane" else 0
        X_test.loc[0, "situ_3.0"] = 1 if situation == "on shoulder" else 0
        X_test.loc[0, "situ_4.0"] = 1 if situation == "on sidewalk" else 0
        X_test.loc[0, "situ_6.0"] = 1 if situation == "on special lane" else 0
        X_test.loc[0, "situ_8.0"] = 1 if situation == "other" else 0

        X_test.loc[0, "populationGrp_Grande Ville"] = (
            1 if population == "big city" else 0
        )
        X_test.loc[0, "populationGrp_Métropole"] = (
            1 if population == "metropolis" else 0
        )
        X_test.loc[0, "populationGrp_Petite Ville"] = (
            1 if population == "small town" else 0
        )
        X_test.loc[0, "populationGrp_Village"] = 1 if population == "village" else 0
        X_test.loc[0, "populationGrp_Ville Moyenne"] = (
            1 if population == "average city" else 0
        )

        X_test.loc[0, "mois_label_aug"] = 1 if mois == "august" else 0
        X_test.loc[0, "mois_label_dec"] = 1 if mois == "december" else 0
        X_test.loc[0, "mois_label_fev"] = 1 if mois == "february" else 0
        X_test.loc[0, "mois_label_jan"] = 1 if mois == "january" else 0
        X_test.loc[0, "mois_label_jul"] = 1 if mois == "july" else 0
        X_test.loc[0, "mois_label_mar"] = 1 if mois == "march" else 0
        X_test.loc[0, "mois_label_oct"] = 1 if mois == "october" else 0

        X_test.loc[0, "etatpGrp_pieton_alone_1"] = 1 if pres_pieton else 0

        X_test.loc[0, "locpGrp_pieton_3_1"] = 1 if loc_pieton else 0

        proba = model_xgboost.predict_proba(X_test)

        limits = [32, 43, 63, 100]
        data_to_plot = (
            "Probability for the accident to be severe",
            round(proba[0, 1] * 100, 0),
        )
        palette = sns.color_palette("Blues", len(limits))
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_yticks([1])
        ax.set_yticklabels([data_to_plot[0]])

        prev_limit = 0
        for idx, lim in enumerate(limits):
            ax.barh(
                [1],
                lim - prev_limit,
                left=prev_limit,
                height=15,
                color=palette[idx],
            )
            prev_limit = lim

        ax.barh([1], data_to_plot[1], color="black", height=5)
        plt.annotate(
            "Transport \n Minister",
            xy=(32, 9),
            xytext=(23, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.annotate(
            "Data \n Expert",
            xy=(43, 9),
            xytext=(40, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.annotate(
            "Economy \n Minister",
            xy=(63, 9),
            xytext=(58, 14),
            arrowprops={
                "facecolor": "blue",
                "linewidth": 0.5,
                "headwidth": 10,
                "headlength": 4,
            },
            size=6,
        )
        plt.tight_layout()
        col95, col96, col97 = st.columns([1, 4, 1])

        with col95:
            st.write(" ")

        with col96:
            st.pyplot(fig=fig)

        with col97:
            st.write(" ")


def page_model_ML():
    st.image("img\Image1.png")
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Modélisation de la gravité avec le Machine Learning</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Cas d'usage métier et objectifs",
         "Approche utilisée",
         "Baseline",
         "Tuning des Hyperparamètres",
         "Entraînement et évaluation finale",
         "Interprétabilité",
         "Démonstration"]
      )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )
    
    with tab3:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab5:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab6:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab7:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )



def tab_demo_LSTM():

    st.markdown(
        "<h2 style='text-align: center; color: black;'>Démonstration </h2>",
        unsafe_allow_html=True,
    )

    st.write(
        """ 
            In this part, we let you choose the parameters.
            \n \n
        """
    )

    st.markdown(
        "<h4 style='text-align: center; color: black;'>Paramètres</h4>",
        unsafe_allow_html=True,
    )


    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        n_days = st.select_slider("Choisissez la nombre de jours de prédiction",
                                    options = [120, 150, 180, 360])
    with col2:
        class_grav = st.selectbox(
            "Classe de la gravité",
            [
                "Toutes les classes confondues (par défaut)",
                "Indemme et bléssés légers",
                "Blessés grave et décédes dans les 30 jours"
            ]
            )


    result = st.button("Predict")
    if result:
        #st.write("Nombre de jours de prédiction:",n_days)
        predict(n_days)

def page_model_DeepLearningLSTM():
    st.image("img\Image1.png")
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Modélisation du nombre d'accidents avec le Deep Learning LSTM</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Cas d'usage métier et objectifs",
         "Approche utilisée",
         "Baseline",
         "Tuning des Hyperparamètres",
         "Entraînement et évaluation finale",
         "Interprétabilité",
         "Démonstration"]
      )

    with tab1:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )
    
    with tab3:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab5:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab6:
        st.markdown(
            "<h4 style='text-align: center; color: black;'></h4>",
            unsafe_allow_html=True,
        )

    with tab7:
        tab_demo_LSTM()


#################################################################################################
# les pages
#################################################################################################
page = st.sidebar.radio(
    "",
    [
        "Introduction",
        "Analyse exploiratoire des données",
        "Modélisation de la gravité",
        "Modélisation du nombre d'accidents",
        "Conclusion",
    ],
)


match page:    
    case "Introduction":
        page_introduction()
    case "Analyse exploiratoire des données":
        page_EAD()
    case "Modélisation de la gravité":
        page_model_ML()
    case "Modélisation du nombre d'accidents":
        page_model_DeepLearningLSTM()
    case "Conclusion":
        page_conclusion()
    case _:
        print("ERROR-page value is not defined:",page)


st.divider()
st.sidebar.info(
    """ Vincent KOMIWES,
       Sidney HANIFA,           
       Farah VARSALLY,
       Simon-Pierre SILGA
       Mentor : Eliott DOUIEB
    """
)

