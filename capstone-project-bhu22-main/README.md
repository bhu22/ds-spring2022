# capstone-project-bhu22
capstone-project-bhu22 created by GitHub Classroom

## Data Science Wine Quality Project 
This project aims to create a web app that is based on our interest in wine to perform a prediction algorithm using 11 input variables (based on the physicochemical tests) to classify wine into three quality groups. This app contains the necessary files (data description, data visualization and machine learning model python files) for the app to run with each of the files that generates a web page and to be integrated to the overall app file. 

The data description file only needs the user to click the button to show the dataset basic information and the input attribute description is included in the presented dataframe. The data source and the citation are also provided for the references.  The data visualization file takes the choice between the white and red wine to show the different plots including the correlation figure of the dataset,  the histogram of each feature, the wine quality distribution and the box plot of quality score versus each feature. 

The machine learning python file takes input which, besides the choice between red or white wine, chooses numbers on the slide bar for the 11 input features and clicks the prediction button to see the outcome sentence based on the prediction of input data. In addition, the performance of random forest models for each dataset respectively are evaluated using the f1 score on the test dataset. The models are saved in pickle format after being trained with 70% of each dataset and the parameters are tuned to reach a better f1 score. As a graphic output, we plot the importance of the features and the display of the confusion matrix. Predicted labels can be compared with true labels for both test and train sets using ternary plots.  

The final app file combines the three separate files (in folder called pages) and main app.py file to choose pages and display the graphs and output accordingly. 
The app can run locally with streamlit with the packages imported such as pickle for loading models, pandas, numpy, matplotlib, sklearn and plotly to allow the smooth output. 
The hosted video links are posted below and this app is really easy to use.  Have fun! Multipage.py is referred from https://github.com/prakharrathi25/data-storyteller . 
The app needs to be run using the following command: 

*streamlit run app.py*


The video demo of the app can be obtained from https://drive.google.com/file/d/1sBUvQJCMVglXQa4sK_6z1YMP-7fY79pN/view?usp=sharing

My contribution to the project:
Beini Hu: data preprocessing, visualization, model training and testing on red wine dataset, streamlit data visualization, ternary plots and ml model prediction. 
