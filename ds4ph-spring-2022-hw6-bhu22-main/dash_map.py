import dash_leaflet as dl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import os

app = Dash()
app.layout = html.Div([
        html.H1('Map Generator'),
        html.Div([
            html.H2('Enter latitude and longitude of a location'),
            dcc.Input(id='latitude', value=39.296941897604654, type='number'),
            dcc.Input(id='longitude', value=-76.59265037135602, type='number'),
            dcc.Input(id='zoom', value=11, type='number'),
            html.Button('Submit', id='submit-button'),
            dl.Map(id='map')
        ])
])

@app.callback(Output("map", "children"),
              Input('submit-button', 'n_clicks'),
              State("latitude", "value"),
              State("longitude", "value"),
              State("zoom", "value"))


def map_click(n_clicks, latitude, longitude, zoom):
    if n_clicks is not None:
        return [dl.Map(
               center=(latitude, longitude),
               zoom=zoom,
               style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"},
               children=[dl.TileLayer(),
                         dl.Marker(position=[latitude, longitude],
                                   children=dl.Tooltip("Here is your location: ({:.3f}, {:.3f})".format(latitude, longitude)))]
        )]

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
