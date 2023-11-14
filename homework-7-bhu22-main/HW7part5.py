from dash import Dash, dcc, html, Input, Output
import os
import numpy as np
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Try Out this Logistic Curve"),
    html.Div([
        html.H2('Choose a beta0 value'),
        dcc.Slider(min=0, max=20, step=1, value=10, id='beta0'),
        html.H2('Choose a beta1 value'),
        dcc.Slider(min=0, max=20, step=1, value=5, id='beta1')
    ]),
    dcc.Graph(id='output-graph')

])


@app.callback(
    Output(component_id='output-graph', component_property='figure'),
    Input(component_id='beta0', component_property='value'),
    Input(component_id='beta1', component_property='value')
)

def update_output_fig(beta0, beta1):
    x = np.linspace(-10,10,1000)
    y = np.exp(beta0+beta1*x)/ (1+ np.exp(beta0+beta1*x))
    fig = px.line(x=x,y=y,title = 'Logistic Regression Curve', labels=dict(y='f(x)'))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')