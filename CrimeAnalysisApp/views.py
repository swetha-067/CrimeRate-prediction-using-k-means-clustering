from django.shortcuts import render
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Global models and preprocessing tools
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
cluster_scaler = MinMaxScaler(feature_range=(0, 1))
cluster_model = None
theft_model = None
murder_model = None
rape_model = None
cluster_high_label = None
TRAINED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset', 'Dataset.csv')


def load_dataset(file_obj=None):
    path = None
    if file_obj is not None:
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_csv(TRAINED_DATA_PATH)
    df.fillna(0, inplace=True)
    return df


def train_models(df):
    global cluster_model, theft_model, murder_model, rape_model, cluster_high_label

    cluster_features = df[['States/UTs', 'District', 'Murder', 'Rape', 'Theft', 'Dowry_Deaths', 'Year']].copy()
    cluster_features['States/UTs'] = state_encoder.fit_transform(cluster_features['States/UTs'].astype(str))
    cluster_features['District'] = district_encoder.fit_transform(cluster_features['District'].astype(str))
    X_cluster = cluster_scaler.fit_transform(cluster_features.values)

    cluster_model = KMeans(n_clusters=2, n_init=50, random_state=42)
    cluster_model.fit(X_cluster)

    center_scores = []
    for center in cluster_model.cluster_centers_:
        center_scores.append(center[2] + center[3] + center[4] + center[5])
    cluster_high_label = int(np.argmax(center_scores))

    future_features = df[['States/UTs', 'District', 'Year', 'Theft', 'Murder', 'Rape']].copy()
    future_features['States/UTs'] = state_encoder.transform(future_features['States/UTs'].astype(str))
    future_features['District'] = district_encoder.transform(future_features['District'].astype(str))
    X_future = future_features[['States/UTs', 'District', 'Year']].values

    theft_model = RandomForestRegressor(n_estimators=100, random_state=42)
    murder_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rape_model = RandomForestRegressor(n_estimators=100, random_state=42)

    theft_model.fit(X_future, future_features['Theft'].values)
    murder_model.fit(X_future, future_features['Murder'].values)
    rape_model.fit(X_future, future_features['Rape'].values)

    return theft_model, murder_model, rape_model


# Train models at import time using the provided dataset file
try:
    base_df = load_dataset()
    theft_model, murder_model, rape_model = train_models(base_df)
except Exception:
    theft_model = murder_model = rape_model = None
    cluster_model = None
    cluster_high_label = None


def UploadDatasetAction(request):
    if request.method == 'POST':
        myfile = request.FILES.get('t1', None)
        if myfile:
            df = load_dataset(myfile)
        else:
            df = load_dataset()

        train_models(df)

        columns = list(df.columns)
        strdata = '<table border=1 align=center width=100%><tr>'
        for column in columns:
            strdata += '<th><font color="black">' + str(column) + '</th>'
        strdata += '</tr>'

        for row in df.values:
            strdata += '<tr>'
            for value in row:
                strdata += '<td><font color="black">' + str(value) + '</td>'
            strdata += '</tr>'
        strdata += '</table>'

        context = {'data': strdata}
        return render(request, 'ViewDataset.html', context)


def AdminLogin(request):
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == 'admin' and password == 'admin':
            return render(request, 'AdminScreen.html', {'data': user})
        return render(request, 'Admin.html', {'data': 'Invalid login details'})


def index(request):
    return render(request, 'index.html', {})


def Admin(request):
    return render(request, 'Admin.html', {})


def UploadDataset(request):
    return render(request, 'UploadDataset.html', {})


def ClusterPrediction(request):
    df = load_dataset()
    df = df[['States/UTs', 'District', 'Year']].copy()
    states = sorted(df['States/UTs'].unique())
    years = sorted(df['Year'].unique())

    output = '<tr><td><font color="black">States</font></td><td><select name="t1">'
    for state in states:
        output += '<option value="{}">{}</option>'.format(state, state)
    output += '</select></td></tr>'

    output += '<tr><td><font color="black">District</font></td><td><select name="t2">'
    for state in states:
        districts = df[df['States/UTs'] == state]['District'].unique()
        output += '<option value="">--{}--</option>'.format(state)
        for district in districts:
            output += '<option value="{}">{}</option>'.format(district, district)
    output += '</select></td></tr>'

    output += '<tr><td><font color="black">Year</font></td><td><select name="t3">'
    for year in years:
        output += '<option value="{}">{}</option>'.format(year, year)
    output += '</select></td></tr>'

    return render(request, 'ClusterPrediction.html', {'states': output})


def ClusterPredictionAction(request):
    if request.method == 'POST':
        state = request.POST.get('t1', '')
        district = request.POST.get('t2', '')
        year = int(request.POST.get('t3', 0))
        murder = float(request.POST.get('t4', 0) or 0)
        rape = float(request.POST.get('t5', 0) or 0)
        theft = float(request.POST.get('t6', 0) or 0)
        dowry = float(request.POST.get('t7', 0) or 0)

        if cluster_model is None:
            return render(request, 'index.html', {'data': 'Models are not trained yet. Upload the dataset first.'})

        test = pd.DataFrame([[state, district, murder, rape, theft, dowry, year]],
                            columns=['States/UTs', 'District', 'Murder', 'Rape', 'Theft', 'Dowry_Deaths', 'Year'])
        test.fillna(0, inplace=True)
        test['States/UTs'] = state_encoder.transform(test['States/UTs'].astype(str))
        test['District'] = district_encoder.transform(test['District'].astype(str))
        X_test = cluster_scaler.transform(test.values)
        label = int(cluster_model.predict(X_test)[0])
        cluster_name = 'High Crime Rate Area' if label == cluster_high_label else 'Low Crime Rate Area'
        output = '{} in {} for {} is predicted as <b>{}</b>'.format(district, state, year, cluster_name)
        return render(request, 'index.html', {'data': output})


def FuturePrediction(request):
    df = load_dataset()
    df = df[['States/UTs', 'District', 'Year']].copy()
    states = sorted(df['States/UTs'].unique())
    years = sorted(df['Year'].unique())

    output = '<tr><td><font color="black">States</font></td><td><select name="t1">'
    for state in states:
        output += '<option value="{}">{}</option>'.format(state, state)
    output += '</select></td></tr>'

    output += '<tr><td><font color="black">District</font></td><td><select name="t2">'
    for state in states:
        districts = df[df['States/UTs'] == state]['District'].unique()
        output += '<option value="">--{}--</option>'.format(state)
        for district in districts:
            output += '<option value="{}">{}</option>'.format(district, district)
    output += '</select></td></tr>'

    output += '<tr><td><font color="black">Year</font></td><td><select name="t3">'
    for year in years:
        output += '<option value="{}">{}</option>'.format(year, year)
    output += '<option value="2027">2027</option>'
    output += '<option value="2028">2028</option>'
    output += '</select></td></tr>'

    return render(request, 'FuturePrediction.html', {'states': output})


def FuturePredictionAction(request):
    if request.method == 'POST':
        state = request.POST.get('t1', '')
        district = request.POST.get('t2', '')
        year = int(request.POST.get('t3', 0))
        classify_type = request.POST.get('t4', 'Theft')

        if theft_model is None or murder_model is None or rape_model is None:
            return render(request, 'index.html', {'data': 'Models are not trained yet. Upload the dataset first.'})

        test = pd.DataFrame([[state, district, year]], columns=['States/UTs', 'District', 'Year'])
        test.fillna(0, inplace=True)
        test['States/UTs'] = state_encoder.transform(test['States/UTs'].astype(str))
        test['District'] = district_encoder.transform(test['District'].astype(str))
        X_test = test.values

        if classify_type == 'Theft':
            predicted = int(round(theft_model.predict(X_test)[0]))
            output = 'Future predicted thefts for {}, {} in {} = {}'.format(district, state, year, predicted)
        elif classify_type == 'Murder':
            predicted = int(round(murder_model.predict(X_test)[0]))
            output = 'Future predicted murders for {}, {} in {} = {}'.format(district, state, year, predicted)
        else:
            predicted = int(round(rape_model.predict(X_test)[0]))
            output = 'Future predicted rapes for {}, {} in {} = {}'.format(district, state, year, predicted)

        return render(request, 'index.html', {'data': output})


def Analysis(request):
    return render(request, 'Analysis.html', {})


def AnalysisAction(request):
    if request.method == 'POST':
        classify_type = request.POST.get('t1', False)
        strdata = '<table border=1 align=center width=100%>'
        if classify_type == 'Theft':
            strdata += '<tr><td><img src="static/analysis/theft_bar.png" height="300" width="500"/></td></tr>'
            strdata += '<tr><td><img src="static/analysis/theft_pie.png" height="300" width="500"/></td></tr>'
        if classify_type == 'Murder':
            strdata += '<tr><td><img src="static/analysis/murder_bar.png" height="300" width="500"/></td></tr>'
            strdata += '<tr><td><img src="static/analysis/murder_pie.png" height="300" width="500"/></td></tr>'
        if classify_type == 'Rape':
            strdata += '<tr><td><img src="static/analysis/rape_bar.png" height="300" width="500"/></td></tr>'
            strdata += '<tr><td><img src="static/analysis/rape_pie.png" height="300" width="500"/></td></tr>'
        return render(request, 'ViewGraphs.html', {'data': strdata})




    
    
