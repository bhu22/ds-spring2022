import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import pickle
import streamlit as st


# Get dataset information based on wine type
def get_dataset(wine_type):
    if wine_type == "Red Wine":
        df = pd.read_csv('winequality-red.csv', sep=';')
        df['wine_label'] = [2 if x > 6 else 1 if ((x > 4) and (x < 7)) else 0.0 for x in df['quality']]
        X = df.drop(['quality', 'wine_label'], axis=1)
        y = df['wine_label']
    elif wine_type == "White Wine":
        df = pd.read_csv('winequality-white.csv', sep=';')
        df['wine_label'] = [2 if x > 6 else 1 if ((x > 4) and (x < 7)) else 0.0 for x in df['quality']]
        X = df.drop(['quality', 'wine_label'], axis=1)
        y = df['wine_label']
    else:
        pass
    return df, X, y


# Load Model
def get_model(wine_type):
    if wine_type == "Red Wine":
        pickle_in = open('RF_redwine.pkl', 'rb')
        classifier = pickle.load(pickle_in)
    elif wine_type == "White Wine":
        pickle_in = open('whitewine.pkl', 'rb')
        classifier = pickle.load(pickle_in)
    else:
        pass
    return classifier


# Split Train and Test dataset
def train_test(X, y, random):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=random)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    # print('X_train:', X_train.shape, 'X_test:', X_test.shape)
    # print('y_train:', y_train.shape, 'y_test:', y_test.shape)
    return X_train, X_test, y_train, y_test


def prediction(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
               total_sulfur_dioxide,
               density, pH, sulphates, alcohol, classifier):
    prediction = classifier.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                      free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    return prediction


def app():
    sns.set()
    st.title('Machine Learning Model Prediction for Wine Dataset')

    # Select Wine type
    wine_type = st.selectbox("Select a type of wine:", ("White Wine", "Red Wine"))
    st.write('You selected:', wine_type)

    df, X, y = get_dataset(wine_type)
    classifier = get_model(wine_type)
    X_train, X_test, y_train, y_test = train_test(X, y, 20)

    # Get y_test pred and y_train pred
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)
    test_result_proba = pd.DataFrame(y_test_proba, columns=['Poor', 'Average', 'Good'])

    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    train_result_proba = pd.DataFrame(y_train_proba, columns=['Poor', 'Average', 'Good'])

    st.subheader(f'Test your {wine_type} quality:')
    # get user input
    fixed_acidity = st.slider("fixed acidity", 3.0, 15.0, 1.0)
    volatile_acidity = st.slider("volatile_acidity", 0.08, 1.1, 0.02)
    citric_acid = st.slider("citric_acid", 0.0, 1.66, 0.01)
    residual_sugar = st.slider("residual_sugar", 0.6, 65.8, 0.2)
    chlorides = st.slider("chlorides", 0.009, 0.346, 0.001)
    free_sulfur_dioxide = st.slider("free_sulfur_dioxide", 2, 289, 1)
    total_sulfur_dioxide = st.slider("total_sulfur_dioxide", 9, 440, 1)
    density = st.slider("density", 0.98, 1.05, 0.01)
    pH = st.slider("pH", 1.0, 4.5, 0.02)
    sulphates = st.slider("sulphates", 0.22, 1.08, 0.02)
    alcohol = st.slider("alcohol", 5.0, 15.0, 0.2)
    # predict output
    if st.button("Predict"):
        result = prediction(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol, classifier)
        if result == 0:
            st.success('Based on our prediction, the quality of this wine could be Poor')
        elif result == 1:
            st.success('Based on our prediction, the quality of this wine could be Average')
        else:
            st.success('Based on our prediction, the quality of this wine could be Good')

    scorepred = f1_score(y_test, y_test_pred, average='weighted')
    st.write(f"f1_score of testing set for {wine_type} model:", scorepred)


    st.subheader('Model plots')
    st.write(f'Ternary Plots of Testing and Training set for {wine_type}:')

    def terplot_prep(y_test_pred, y_test, test_result_proba):
        test_pred = pd.Series(y_test_pred, name='Predicted')
        true_label_test = y_test.rename('True Label').reset_index()
        test_df = pd.concat([true_label_test, test_pred], axis=1)
        test_df = test_df.drop('index', axis=1)
        test_plot = pd.DataFrame(columns=['True_Value', 'True_Label', 'Predicted', 'Poor', 'Average', 'Good',
                                          'Predicted Same as True Label'])
        test_plot['True_Value'] = test_df['True Label']
        test_plot['True_Label'] = test_df['True Label']
        test_plot = test_plot.replace({'True_Label': {0: 'Poor', 1: 'Average', 2: 'Good'}})
        test_plot['Predicted'] = test_df['Predicted']
        test_plot['Average'] = test_result_proba['Average']
        test_plot['Poor'] = test_result_proba['Poor']
        test_plot['Good'] = test_result_proba['Good']
        test_plot['Predicted Same as True Label'] = np.where(test_df['Predicted'] == test_df['True Label'], 'Yes', 'No')
        return test_plot

    test_plot = terplot_prep(y_test_pred, y_test, test_result_proba)
    train_plot = terplot_prep(y_train_pred, y_train, train_result_proba)

    colors_test = {
        "Poor": "#0C3B5D",
        "Average": "#3EC1CD",
        "Good": "#EF3A4C",
    }
    fig = px.scatter_ternary(test_plot, a="Poor", b="Average", c="Good", color='True_Label',
                             color_discrete_map=colors_test, symbol="Predicted Same as True Label",
                             symbol_map={'Yes': 'circle-open', 'No': 'x'},
                             width=800,
                             height=600, title='Ternary plot of True and Predicted Outcomes for Testing Set')

    st.plotly_chart(fig)

    colors_train = {
        "Poor": "#3cb371",
        "Average": "#ffa500",
        "Good": "#6a5acd",
    }
    fig = px.scatter_ternary(train_plot, a="Poor", b="Average", c="Good", color='True_Label',
                             color_discrete_map=colors_train, symbol="Predicted Same as True Label",
                             symbol_map={'Yes': 'circle-open', 'No': 'x'},
                             width=800,
                             height=600, title='Ternary plot of True and Predicted Outcomes for Training Set')

    st.plotly_chart(fig)

    # plot feaure importance of the model
    st.write(f'Feature Importance plot of {wine_type} model:')
    fig = plt.figure()
    feat_importances = pd.Series(classifier.feature_importances_,
                                 index=df.drop(['quality', 'wine_label'], axis=1).columns).sort_values()
    feat_importances.plot(kind='barh')
    st.pyplot(fig)

    # Plot confusion matrix
    st.write(f'Confusion Matrix Plots for Train and Test dataset of {wine_type} :')

    class_names = ['Poor', 'Average', 'Good']

    cnf_matrix_train = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_train, display_labels=class_names)
    disp.plot(cmap='Blues_r')
    plt.grid(False)
    plt.title('Confusion Matrix for Training Dataset')
    st.pyplot(plt.gcf())

    cnf_matrix_test = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_test, display_labels=class_names)
    disp.plot(cmap='Blues_r')
    plt.grid(False)
    plt.title('Confusion Matrix for Testing Dataset')
    st.pyplot(plt.gcf())

