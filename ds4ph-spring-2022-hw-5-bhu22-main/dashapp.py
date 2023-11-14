from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import requests as rq
import bs4

url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
page = rq.get(url)

bs4page = bs4.BeautifulSoup(page.text, 'html.parser')
tables = bs4page.find('table',{'class':"wikitable"})

gdp_country = pd.read_html(str(tables), header=[1])[0]
gdp_country = gdp_country.dropna()

gdp_country.columns = ['Country/Territory', 'Region', 'IMF_Estimate', 'IMF_Year', 'UN_Estimate', 'UN_Year', 'WB_Estimate', 'WB_Year']
gdp_country = gdp_country[['Country/Territory', 'Region', 'IMF_Estimate', 'UN_Estimate',  'WB_Estimate']]

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(options=[
            {'label': 'IMF', 'value': 1},
            {'label': 'United Nations', 'value': 2},
            {'label': 'World Bank', 'value': 3}
        ],
        value=1, id='input-level'
                ),
    dcc.Graph(id='output-graph')
])

@app.callback(
    Output('output-graph', 'figure'),
    Input('input-level', 'value'))

def update_figure(selected_org):
    if int(selected_org) == 1:
        gdp_df = gdp_country[['Country/Territory', 'Region', 'IMF_Estimate']]
        fig = px.bar(gdp_df, x="Region", y= "IMF_Estimate", color="Country/Territory",
             labels={
                 "IMF_Estimate":"GDP (US$)"},
                 title="GDP Estimated by IMF")
    elif int(selected_org) == 2:
        gdp_df = gdp_country[['Country/Territory', 'Region', 'UN_Estimate']]
        fig = px.bar(gdp_df, x="Region", y="UN_Estimate", color="Country/Territory",
             labels={
                 "UN_Estimate":"GDP (US$)"},
                 title="GDP Estimated by United Nations")
    else:
        gdp_df = gdp_country[['Country/Territory', 'Region', 'WB_Estimate']]
        fig = px.bar(gdp_df, x="Region", y="WB_Estimate", color="Country/Territory",
             labels={
                 "WB_Estimate":"GDP (US$)"},
                 title="GDP Estimated by World Bank")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')