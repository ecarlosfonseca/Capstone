import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
import numpy as np


########################################
# Begin database stuff


DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)
    predicted_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions

def check_request(request):
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error

    return True, ""



def check_valid_column(observation):
    
    valid_columns = {
        "Type",
        "Date",
        "Part of a policing operation",
        "Latitude",
        "Longitude",
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "station"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "The following columns are missing: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "These columns are not expected: {}".format(extra)
        return False, error    

    return True, ""



def check_categories(observation):
    
    category_map = {
        "Type": ["Person search", "Person and Vehicle search", "Vehicle search"],
        "Part of a policing operation": [True, False],
        "Gender": ["Male", "Female"],
        "Age range": ["10-17", "18-24", "25-34", "over 34"],
        "Officer-defined ethnicity": ["Asian", "White", "Black", "Other", "Mixed"],
        "Legislation": ['Misuse of Drugs Act 1971 (section 23)', 
                        'Police and Criminal Evidence Act 1984 (section 1)', 
                        'Psychoactive Substances Act 2016 (s36(2))', 
                        'Criminal Justice Act 1988 (section 139B)',
                        'Firearms Act 1968 (section 47)',
                        'Poaching Prevention Act 1862 (section 2)',
                        'Criminal Justice and Public Order Act 1994 (section 60)',
                        'Police and Criminal Evidence Act 1984 (section 6)', 
                        'Wildlife and Countryside Act 1981 (section 19)', 
                        'Psychoactive Substances Act 2016 (s37(2))', 
                        'Aviation Security Act 1982 (section 27(1))', 
                        'Protection of Badgers Act 1992 (section 11)',
                        'Crossbows Act 1987 (section 4)',  
                        'Public Stores Act 1875 (section 6)',
                        'Customs and Excise Management Act 1979 (section 163)',
                        'Deer Act 1991 (section 12)',
                        'Conservation of Seals Act 1970 (section 4)'],
        "station": ['avon-and-somerset',
                     'bedfordshire',
                     'btp',
                     'cambridgeshire',
                     'cheshire',
                     'city-of-london',
                     'cleveland',
                     'cumbria',
                     'derbyshire',
                     'devon-and-cornwall',
                     'dorset',
                     'durham',
                     'dyfed-powys',
                     'essex',
                     'gloucestershire',
                     'greater-manchester',
                     'gwent',
                     'hampshire',
                     'hertfordshire',
                     'humberside',
                     'kent',
                     'lancashire',
                     'leicestershire',
                     'lincolnshire',
                     'merseyside',
                     'metropolitan',
                     'norfolk',
                     'north-wales',
                     'north-yorkshire',
                     'northamptonshire',
                     'northumbria',
                     'nottinghamshire',
                     'south-yorkshire',
                     'staffordshire',
                     'suffolk',
                     'surrey',
                     'sussex',
                     'thames-valley',
                     'warwickshire',
                     'west-mercia',
                     'west-yorkshire',
                     'wiltshire']
    }
    
    for key, valid_categories in category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""

# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
    
    obs_dict = request.get_json()
    _id = obs_dict['observation_id']
    observation = obs_dict
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
  
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'error': error}
        return jsonify(response)

    _id = obs_dict['observation_id']
    observation = obs_dict
    del observation['observation_id']

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categories(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    proba = pipeline.predict_proba(obs)[0, 1]
    pred = pipeline.predict(obs)[0]
    if proba >= 0.20069490313711058:
        predicted_class=True
    else:
        predicted_class=False

    response = {'outcome': bool(pred)}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    
    try:
        p.save()
    except IntegrityError:
        error_msg = 'ERROR: Observation ID: {} already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def search_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        saved_outcome = bool(p.true_class)
        p.true_class = obs['outcome']
        p.save()
        return jsonify({'observation_id': obs['observation_id'],
                        'outcome': bool(obs['outcome']),
                        'predicted_outcome':saved_outcome})
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run()