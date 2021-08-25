import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)

model_obj = joblib.load('booking_cancel.joblib')
model = model_obj['model']

numeric_cols = model_obj['numeric_cols']
encoded_cols = model_obj['encoded_cols']
categorical_cols = model_obj['categorical_cols']


def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df.is_repeated_guest = input_df.is_repeated_guest.astype('object')
    input_df[numeric_cols] = model_obj['imputer'].transform(input_df[numeric_cols])
    input_df[numeric_cols] = model_obj['scaler'].transform(input_df[numeric_cols])
    input_df[encoded_cols] = model_obj['encoder'].transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

@app.route('/')
def home():
    # return render_template('index.html')
    return("<h3>Welcome to Predict booking cancellation API!</h3><p>See README.md for more info.</p>")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    print(data)

    # prediction = model.predict([np.array(list(data.values()))])
    pred, prob = predict_input(model, data)
    print(pred, prob)
    return jsonify(int(pred), prob)
    




if __name__ == "__main__":
    app.run(debug=True)