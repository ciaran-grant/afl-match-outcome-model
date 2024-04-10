import numpy as np
import pandas as pd
import joblib
from flask import Flask, request
from AFLPy.AFLData_Client import load_data, upload_data
from AFLPy.AFLBetting import submit_tips, get_current_tips
from afl_match_outcome_model.predict.predict_outcome import load_outcome_model, get_outcome_prediction
from afl_match_outcome_model.predict.predict_margin import load_margin_model, load_margin_preprocessor
from afl_match_outcome_model.data_preparation.update_preprocessor import update_fit_new_expected_data, update_fit_new_squads
from afl_match_outcome_model.data_preparation.match_id_utils import get_home_team_from_match_id, get_away_team_from_match_id
app = Flask(__name__)

@app.route("/model/outcome/predict", methods=["GET", "POST"])
def predict_outcome(ID = None):
    
    model = load_outcome_model()
    model_features = model.xgb_model.get_booster().feature_names
        
    data = load_data(Dataset_Name="Match_Stats_Enriched", ID = request.json['ID'])      
    
    data = get_outcome_prediction(data, model, model_features)
    
    upload_data(Dataset_Name="CG_Match_Outcome", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')

@app.route("/model/margin/update_expected_data", methods=["GET", "POST"])
def update_expected_data(ID = None):
    
    update_fit_new_expected_data(ID = request.json['ID'])
    
    return [True]

@app.route("/model/margin/update_squads", methods=["GET", "POST"])
def update_squads(ID = None):
    
    update_fit_new_squads(ID = request.json['ID'])
    
    return [True]

@app.route("/model/margin/preprocess", methods=["GET", "POST"])
def preprocess_margin(ID = None):
    
    match_summary = load_data(Dataset_Name="AFL_API_Matches", ID = request.json['ID']).sort_values(by = "Match_ID", ascending = True).reset_index(drop = True)

    preproc = load_margin_preprocessor()
    preprocessed_data = preproc.transform(match_summary)
    preprocessed_data['Match_ID'] = match_summary['Match_ID']
    preprocessed_data = preprocessed_data[['Match_ID'] + [x for x in list(preprocessed_data) if x != 'Match_ID']]
    
    upload_data(Dataset_Name="CG_Margin_Features", Dataset=preprocessed_data, overwrite=True, update_if_identical=True)

    return preprocessed_data.to_json(orient='records')

@app.route("/model/margin/predict", methods=["GET", "POST"])
def predict_margin(ID = None):
    
    data = load_data(Dataset_Name="CG_Margin_Features", ID = request.json['ID']).sort_values(by = "Match_ID", ascending = True).reset_index(drop = True)
    
    model = load_margin_model()
    model_features = model.xgb_model.get_booster().feature_names
    data[model_features] = data[model_features].apply(pd.to_numeric, axis=1)

    data['Predicted_Margin'] = model.predict(data[model_features])

    upload_data(Dataset_Name="CG_Match_Margin", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')

@app.route("/model/createtipping", methods=["GET", "POST"])
def create_tipping(ID = None):
    
    # Call predict match margin
    tipping_data = load_data(Dataset_Name="CG_Match_Margin", ID = request.json['ID'])
    tipping_data['Home_Team'] = tipping_data['Match_ID'].apply(lambda x: get_home_team_from_match_id(x))
    tipping_data['Away_Team'] = tipping_data['Match_ID'].apply(lambda x: get_away_team_from_match_id(x))
    tipping_data['Predicted_Team'] = np.where(tipping_data['Predicted_Margin'] > 0, tipping_data['Home_Team'], tipping_data['Away_Team'])
    tipping_data = tipping_data[['Match_ID', 'Predicted_Team', 'Predicted_Margin']]
    tipping_data['Predicted_Margin'] = abs(tipping_data['Predicted_Margin'].astype(int))
    
    upload_data(Dataset_Name="CG_Tipping", Dataset=tipping_data, overwrite=True, update_if_identical=True)
    
    return tipping_data.to_json(orient='records')


@app.route("/model/tipping", methods=["GET", "POST"])
def apply_tipping(ID = None):
    
    data = load_data(Dataset_Name="CG_Tipping", ID = request.json['ID'])
    submit_tips(data)
    
    return get_current_tips(ID = request.json['ID'])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
    
