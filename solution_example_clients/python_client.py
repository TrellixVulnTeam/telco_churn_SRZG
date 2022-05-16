# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import base64
import json
import os
import numpy as np
import requests


# Define the input sample.
sample_dict = {
    'customer_ID': '3668-QPYBK',
    'gender': 'Male',
    'has_dependents': 'No',
    'has_device_protection': 'No',
    'has_multiple_lines': 'No',
    'has_online_backup': 'Yes',
    'has_online_security': 'Yes',
    'has_paperless_billing': 'Yes',
    'has_partner': 'No',
    'has_phone_service': 'Yes',
    'has_streaming_movies': 'No',
    'has_streaming_ts': 'No',
    'has_tech_support': 'No',
    'monthly_charges': 53.85,
    'senior_citizen': 0,
    'tenure_months': 2,
    'total_charges': 108.15,
    'type_of_contract': 'Month-to-month',
    'type_of_internet_service': 'DSL',
    'type_of_payment_method': 'Mailed check'
}

# Make a request to the solution server.
url = 'http://127.0.0.1:5001/predict'
headers = {'Content-type': 'application/json'}
body = str.encode(json.dumps({"data": [sample_dict]}))
response = requests.post(url, body, headers=headers)

print(response.json())

