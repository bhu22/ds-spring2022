import streamlit as st
import pandas as pd

def app():
    st.markdown("## Data Science Wine Quality Project Dataset Description")
    st.write("You're presented with each features to determine the wine quality. Click below! ")
    btn = st.button("Click!")
    if btn:
        st.markdown("## Datasets: ")
        st.write("The datasets are used to test classification (multi-class) task, which includes two datasets: winequality-red.csv - red wine preference samples; winequality-white.csv - white wine preference samples; http://www3.dsi.uminho.pt/pcortez/wine/ ")
        st.write("Vinho verde is a unique product from the Minho (northwest) region of Portugal. Medium in alcohol, is it particularly appreciated due to its freshness (specially in the summer). More details can be found at: http://www.vinhoverde.pt/en/")

        st.markdown("## Data input variables (based on physicochemical tests) & attributes description:")
        st.write("Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).")
        st.write(pd.DataFrame({
         'Input features ': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
           'Units': ['tartaric acid - g / dm^3', '(g / dm^3)', '(g / dm^3)', '(g / dm^3)','sodium chloride - g / dm^3','(mg / dm^3)','(mg / dm^3)', '(g / cm^3)', 'scale from 0 (very acidic) to 14 (very basic)',
           '(potassium sulphate - g / dm3)','alcohol (% by volume)'  ],
           'Description': [' most acids involved with wine or fixed or nonvolatile (do not evaporate readily)',
         'the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste',
         'found in small quantities, citric acid can add ‘freshness’ and flavor to wines',
         ' the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet',
         ' the amount of salt in the wine',
         'the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine',
         'amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine',
         'the density of water is close to that of water depending on the percent alcohol and sugar content',
         'describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale',
         'a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant',
         'the percent alcohol content of the wine'
    ],
     }))



        st.markdown("## Dataset Source:")
        st.write('Dataset is downloaded on the UCI Machine Learning Repository website: https://archive.ics.uci.edu/ml/datasets/wine+quality')

        st.markdown("## Citation: ")
        st.write("P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.")
        st.write("Image is downloaded from: https://www.liquibox.com/solution/beverage/wine/")

        st.markdown("###### Project done by Beini Hu and Yuecen Jin, ds4ph-2022 ")


        st.balloons()

