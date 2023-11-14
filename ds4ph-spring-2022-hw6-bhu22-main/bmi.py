from dash import Dash, dcc, html, Input, Output
import os

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Enter your data to see the results"),
    html.Div([
        html.H2('Enter your weight in kg or lbs'),
        dcc.Input(id='weight', value=55, type='number'),
        dcc.RadioItems(options=[
            {'label': 'kilograms', 'value': 'kg'},
            {'label': 'pounds', 'value': 'lbs'}],
            value='kg', id='weight_unit'),
        html.H2('Enter your height in meters or feet'),
        dcc.Input(id='height', value=1.6, type='number'),
        dcc.RadioItems(options=[
            {'label': 'meters', 'value': 'm'},
            {'label': 'feet', 'value': 'ft'}],
            value='m', id='height_unit')
    ]),
    html.Br(),
    html.H1("Your BMI Index is: "),
    html.H1(id='bmi'),

])


@app.callback(
    Output(component_id='bmi', component_property='children'),
    Input(component_id='weight', component_property='value'),
    Input(component_id='weight_unit', component_property='value'),
    Input(component_id='height', component_property='value'),
    Input(component_id='height_unit', component_property='value')
)

def update_output_div(weight, weight_unit, height, height_unit):
    if not weight or not height:
        rval = 'Error'
    elif weight_unit == 'kg' and height_unit == 'm':
        rval = weight / (height ** 2)
    elif weight_unit == 'lbs' and height_unit == 'ft':
        rval = 703 * weight / (height * 12)**2
    else:
        rval = 'Please make sure units of weight/height should be kilograms/meters or pounds/feet'

    return rval

if __name__ == '__main__':
    app.run_server(host = 'jupyter.biostat.jhsph.edu', port = os.getuid()+36, debug=True)