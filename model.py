import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib


def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df.is_repeated_guest = input_df.is_repeated_guest.astype('object')
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob



# Load dataset
raw_df = pd.read_csv('hotel-booking-demand/hotel_bookings.csv')

# Prepare data
raw_df.is_repeated_guest = raw_df.is_repeated_guest.astype('object')

# New features
raw_df['total_children'] = raw_df.children + raw_df.babies
raw_df['total_people'] = raw_df.total_children + raw_df.adults

#Splitting Training, Validation and Test Set
train_df = raw_df[raw_df.arrival_date_year < 2017]
val_df = raw_df[(raw_df.arrival_date_year == 2017) & (raw_df.arrival_date_week_number <= 19)]
test_df = raw_df[(raw_df.arrival_date_year == 2017) & (raw_df.arrival_date_week_number > 19)]


# Identifying Input and Target Columns
input_cols = ['hotel', 'lead_time', 'arrival_date_year',
            'arrival_date_month', 'arrival_date_week_number',
            'arrival_date_day_of_month', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
            'country', 'market_segment', 'distribution_channel',
            'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'reserved_room_type',
            'assigned_room_type', 'booking_changes', 'deposit_type',
            'days_in_waiting_list', 'customer_type',
            'required_car_parking_spaces', 'total_of_special_requests',
            'total_children', 'total_people']

target_col = 'is_canceled'

# Inputs and Targets
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# Numerical and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# Imputing Missing Numeric Data
imputer = SimpleImputer(strategy='mean')
imputer.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scaling Numeric Features
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# Encoding Categorical Data
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# Set X inputs
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


#Fitting model with trainig data
model = RandomForestClassifier(n_jobs=-1, 
                               random_state=11, 
                               n_estimators=200,
                               max_depth=10,
                               max_leaf_nodes=20,
                               class_weight={0: 1, 1: 2})
model.fit(X_train, train_targets)


# Saving model to disk
booking_cancel = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}

joblib.dump(booking_cancel, 'booking_cancel.joblib')


# Loading model to test the results

booking_cancel2 = joblib.load('booking_cancel.joblib')

test_preds2 = booking_cancel2['model'].predict(X_test)
print('Accuracy score for model:', accuracy_score(test_targets, test_preds2))



# Test New input
print("Making Prediction on one input")

new_input = {'hotel': 'Resort Hotel',
            'lead_time': 1,
            'arrival_date_year': 2017,
            'arrival_date_month': 'July',
            'arrival_date_week_number': 27,
            'arrival_date_day_of_month': 25,
            'stays_in_weekend_nights': 3,
            'stays_in_week_nights': 2,
            'adults': 2,
            'children': 1,
            'babies': 0,
            'meal': 'BB',
            'country': 'PRT',
            'market_segment': 'Direct',
            'distribution_channel': 'Direct',
            'is_repeated_guest': 0,
            'previous_cancellations': 4,
            'previous_bookings_not_canceled': 0,
            'reserved_room_type': 'C',
            'assigned_room_type': 'C',
            'booking_changes': 7,
            'deposit_type': 'No Deposit',
            'days_in_waiting_list': 0,
            'customer_type': 'Transient',
            'adr': 0.0,
            'required_car_parking_spaces': 0,
            'total_of_special_requests': 0,
            'total_children': 0,
            'total_people': 3}

pred, prob = predict_input(model, new_input)
print("Prediction: {}, probability: {}".format(pred, prob))