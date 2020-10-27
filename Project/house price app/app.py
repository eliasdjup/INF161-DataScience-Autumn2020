import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template
from waitress import serve

app = Flask(__name__)

# read and prepare model 
model_imputer = pickle.load(open('model.pkl', 'rb'))
model = model_imputer['model']
imputer = model_imputer['imputer']

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

    # handle wrong input
    def numeric_features(value):
        try:
            return float(value)
        except:
            return np.nan
    features = {key: numeric_features(value) for key, value in features.items()}

    # prepare for prediction
    features_df = pd.DataFrame(features, index=[0]).loc[:, ['size', 'bedrooms', 'floor']]

    # sjekk input
    if features_df.loc[0, 'size'] <= 0:
        return render_template('./index.html',
                               prediction_text='Size must be positive')

    # predict
    imputed_data = imputer.transform(features_df)
    prediction = model.predict(imputed_data)
    prediction = np.round(prediction[0])
    prediction = np.clip(prediction, 0, np.inf)

    # prepare output
    return render_template('./index.html',
                           prediction_text='Size {}, bedrooms {}, floor {}, predicted price {}'.format(
                               imputed_data[0, 0],
                               imputed_data[0, 1],
                               imputed_data[0, 2],
                               prediction))

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
