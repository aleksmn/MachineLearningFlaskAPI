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



# new_input = {'hotel': 'Resort Hotel',
#             'lead_time': 1,
#             'arrival_date_year': 2017,
#             'arrival_date_month': 'July',
#             'arrival_date_week_number': 27,
#             'arrival_date_day_of_month': 25,
#             'stays_in_weekend_nights': 3,
#             'stays_in_week_nights': 2,
#             'adults': 2,
#             'children': 1,
#             'babies': 0,
#             'meal': 'BB',
#             'country': 'PRT',
#             'market_segment': 'Direct',
#             'distribution_channel': 'Direct',
#             'is_repeated_guest': 0,
#             'previous_cancellations': 4,
#             'previous_bookings_not_canceled': 0,
#             'reserved_room_type': 'C',
#             'assigned_room_type': 'C',
#             'booking_changes': 7,
#             'deposit_type': 'No Deposit',
#             'days_in_waiting_list': 0,
#             'customer_type': 'Transient',
#             'adr': 0.0,
#             'required_car_parking_spaces': 0,
#             'total_of_special_requests': 0,
#             'total_children': 0,
#             'total_people': 3}

# r = predict_input(model, new_input)

# print(r)



@app.route('/')
def home():
    return render_template('index.html')


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