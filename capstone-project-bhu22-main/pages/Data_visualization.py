import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
sns.set()

#get dataframe based on wine type
def get_dataframe(wine_type):
    if wine_type == "Red Wine":
        df = pd.read_csv('winequality-red.csv',sep=';')
    elif wine_type == "White Wine":
        df = pd.read_csv('winequality-white.csv', sep=';')
    else:
        pass
    return df

def app():
    st.title('Data Visualization for Wine Dataset')

    # select a wine type
    wine_type = st.selectbox("Select a type of wine:",("White Wine","Red Wine"))
    st.write('You selected:', wine_type)

    df = get_dataframe(wine_type)
    st.write("Shape of dataset", df.iloc[:, :11].shape)
    # Dataset visualization
    st.subheader('Visualize the dataset')
    st.write('Distribution plot of the dataset:')
    fig = plt.figure(figsize=[22, 16])
    cols = df.columns
    cnt = 1
    for col in cols:
        plt.subplot(4, 3, cnt)
        sns.histplot(df[col], kde=True)
        cnt += 1
    st.pyplot(fig)

    st.write('Distribution plot of wine quality score:')
    fig= plt.figure()
    sns.barplot(x=df['quality'].unique(), y=df['quality'].value_counts())
    plt.xlabel("Quality Score")
    plt.ylabel(f"Number of {wine_type}")
    plt.title(f"Distribution of {wine_type} Quality Scores")
    st.pyplot(fig)

    st.write(f'Relationship between {wine_type} quality score and various features :')
    fig, ax1 = plt.subplots(4, 3, figsize=(22, 16))
    k = 0
    for i in range(4):
        for j in range(3):
            if k != 11:
                sns.boxplot(x="quality", y=df.iloc[:, k], data=df, ax=ax1[i][j])
                k += 1
    st.pyplot(fig)

    st.write(f'Correlation map of {wine_type} :')
    fig =plt.figure(figsize=(18, 7))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(fig)













