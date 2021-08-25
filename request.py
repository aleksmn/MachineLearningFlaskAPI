import requests

url = 'http://localhost:5000/predict_api'

# Example of request:

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


r = requests.post(url,json=new_input)

pred, prob = r.json()

print("The prediction is {} with probability {:.2f}.".format(pred, prob))
