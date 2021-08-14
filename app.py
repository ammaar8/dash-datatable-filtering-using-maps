"""
This is a demo file which shows how you can use mapbox scatter plots to 
filter data in dash datatable when you want to be able to filter data
in other columns using native filtering and require row selection.
"""

import dash
import numpy as np
import plotly.express as px
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_html_components as html
import dash_table
import json
import plotly.graph_objects as go

px.set_mapbox_access_token(open(".mapbox_token").read())
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

layout = dbc.Container(dbc.Row([
    dcc.Location(id='url', refresh=False),
    dcc.Store('df'),
    dbc.Col(dbc.Row([
        dbc.Col([
            dcc.Graph(id='map')
        ],
            width=12
        ),
        dbc.Col([
            dash_table.DataTable(
                id='data-table',
                columns=[{'name': str(x), 'id': str(x)} for x in pd.DataFrame(columns=['']).columns],
                row_selectable='multi',
                filter_action='native',
                page_action='native',
                sort_action='native',
                page_size=10,
                page_current=0,
                style_table={'overflowX': 'auto', 'cellspacing': '0', 'width': '100%'},
                # We add the below style to hide the extra column we added for filtering
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'OnMap',},
                        'display': 'None',
                    }
                ],
            ),
        ],
            width=12
        ),
    ])),
]),
    fluid=True,
)


app.layout = layout

@app.callback(
    Output('df', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def load_data(unused):
    df = pd.read_csv('worldcities.csv')
    df = df[['city', 'lat', 'lng', 'country', 'population']]
    df = df.dropna()
    df = df.loc[0:1000, :]
    df = df.sort_values(by='population', ascending=False)
    df = df.to_json()
    if df is None:
        PreventUpdate
    return json.dumps(df)


@app.callback(
    Output('map', 'figure'),
    Input('df', 'data'),
)
def populate_map(df):
    if df is None:
        raise PreventUpdate
    df = json.loads(df)
    df = pd.read_json(df)

    # Map
    map = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lng',
        hover_name="city"
    )
    map.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=df['lat'].median(),
                lon=df['lng'].median()
            ),
            pitch=0,
            # zoom=zoom
        ),
    )
    return map


@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Input('df', 'data'),
    Input('map', 'selectedData'),
    prevent_initial_call=True
)
def populate_table(df, selectedData):
    """
    Callback to populate datatable. This callback also contains 
    logic required to allow filtering by map.
    """
    if df is None:
        raise PreventUpdate
    df = json.loads(df)
    df = pd.read_json(df)
    """
    Map filtering
    How it works ->
    We get the selected rows from the map and create a list of
    rows which are selected on the map. Using this data we create
    a new column which is of boolean type. It is True if the city/point
    is selected else False.
    If no data is selected, the whole columns should be True which is handled
    by the else condition.
    """
    if selectedData is not None:
        selectedCities = []
        if 'points' in selectedData:
            if selectedData['points'] is not None:
                for point in selectedData['points']:
                    selectedCities.append(point['hovertext'])
        df['OnMap'] = df['city'].isin(selectedCities)
    else:
        df['OnMap'] = pd.Series(np.full(len(df['city']), True))
    # Data Table
    data = df.to_dict('records')
    columns = [{'name': str(x), 'id': str(x), "type": 'any', 'selectable': True} for x in df.columns]

    return data, columns


@app.callback(
    Output('data-table', 'filter_query'),
    Input('data-table', 'data'),
    State('data-table', 'filter_query'),
    prevent_initial_call=True
)
def filter_query_for_edit(trigger, query):
    """
    Callback to inject query to allow the table to be filtered 
    by the hidden OnMap column. This callback enures that the 
    map filtering will work even when other column filters are 
    being used.
    """
    if query is None:
        """
        This is called on the first time this callback is fired
        after the datatable is populated. What is essentially does is
        inserts 'contains true' in the filter input box in the column
        for OnMap. 
        """
        query = '{OnMap} contains true'

    if '{OnMap} contains true' not in query:
        """
        This is here to essentially ensure the query is nevery missing
        'contains true' in the OnMap column. Generally it won't be called.
        """
        query += ' && {OnMap} contains true'

    return query

if __name__ == '__main__':
    app.run_server(debug=True)