import random

import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template
from waitress import serve
import pandas as pd

app = Flask(__name__)

# read and prepare model 
model_imputer = pickle.load(open('model.pkl', 'rb'))
model = model_imputer['model']
imputer = model_imputer['imputer']
ratings_matrix = model_imputer['ratings_matrix']
baseline = model_imputer['baseline']
predictions = pd.read_csv('predictions.csv')


# Used to get movie info
movies_df = pd.read_csv ('clean_data/film.csv')


@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    Rendering results on HTML
    '''
    # get data
    features = dict(request.form)

    # prepare for prediction
    features_df = pd.DataFrame(features, index=[0]).loc[:, ['UserID', 'model']]

    user_id = int(features_df.loc[0, 'UserID'])

    if user_id < 0 or user_id > 6040:
        return render_template('./index.html',
                               prediction_text='UserID must be between 0 and 6040')


    if features_df.loc[0, 'model'] == "model2":
        movies = random.sample(baseline.tolist(), 10)
        full_m_df = movies_df[movies_df['FilmID'].isin(movies)]
        reduced_m_df = full_m_df[['Tittel','Aar']]

        return render_template('./index.html',
                        prediction_text="Baseline recommendation for user " + str(user_id)+ "\n\n"+
                        reduced_m_df.to_html(index=False,justify='left'))
    else :
        movies_rated_by_user = ratings_matrix[user_id][ratings_matrix[user_id] != 0].index.values
        predicted_values = predictions.iloc[:,user_id+1]
        i = predicted_values.index.values
        v = predicted_values.values
        l = list(zip(i, v))
        l_sorted = (sorted(l, key=lambda x: x[1], reverse=True))

        res = []

        for movie_id, value in l_sorted:
            if len(res) == 10:
                break
            if (movie_id in movies_rated_by_user):
                continue
            else:
                res.append(movie_id)

        full_m_df = movies_df[movies_df['FilmID'].isin(res)]
        reduced_m_df = full_m_df[['Tittel', 'Aar']]

        return render_template('./index.html',
                           prediction_text="Specific recommendation for user " + str(user_id)+ "\n\n"+
                           reduced_m_df.to_html(index=False, justify='left'))

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
