import requests as rq
import bs4
import pandas as pd
import plotly.express as px

url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
page = rq.get(url)

bs4page = bs4.BeautifulSoup(page.text, 'html.parser')
tables = bs4page.find('table',{'class':"wikitable"})

gdp_country = pd.read_html(str(tables), header=[1])[0]
gdp_country = gdp_country.dropna()

gdp_country.columns = ['Country/Territory', 'Region', 'IMF_Estimate', 'IMF_Year', 'UN_Estimate', 'UN_Year', 'WB_Estimate', 'WB_Year']
gdp_country = gdp_country[['Country/Territory', 'Region', 'IMF_Estimate', 'UN_Estimate',  'WB_Estimate']]

# Dataframes of GDP estimated by different organizations
IMF_GDP = gdp_country[['Country/Territory', 'Region', 'IMF_Estimate']]
UN_GDP = gdp_country[['Country/Territory', 'Region', 'UN_Estimate']]
WB_GDP = gdp_country[['Country/Territory', 'Region', 'WB_Estimate']]

# Plot of GDP Estimated by IMF
fig = px.bar(IMF_GDP, x="Region", y="IMF_Estimate", color="Country/Territory", 
             labels={
                 "IMF_Estimate":"GDP (US$)"},
                 title="GDP Estimated by IMF")
fig.show()

# Plot of GDP Estimated by United Nations
fig = px.bar(UN_GDP, x="Region", y="UN_Estimate", color="Country/Territory", 
             labels={
                 "UN_Estimate":"GDP (US$)"},
                 title="GDP Estimated by United Nations")
fig.show()

# Plot of GDP Estimated by World Bank
fig = px.bar(WB_GDP, x="Region", y="WB_Estimate", color="Country/Territory", 
             labels={
                 "WB_Estimate":"GDP (US$)"},
                 title="GDP Estimated by World Bank")
fig.show()