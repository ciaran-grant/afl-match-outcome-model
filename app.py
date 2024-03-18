from flask import Flask, request
from AFLPy.AFLData_Client import load_data, upload_data
from AFLPy.AFLBetting import submit_tips, get_current_tips
from afl_match_outcome_model.predict.predict_outcome import load_outcome_model, get_outcome_prediction
from afl_match_outcome_model.predict.predict_margin import load_margin_model, get_margin_prediction
from afl_match_outcome_model.data_preparation import create_match_stats_enriched, create_player_stats_enriched

app = Flask(__name__)

@app.route("/preprocess/playerstats", methods=["GET", "POST"])
def create_player_stats(ID = None):
    
    player_stats = create_player_stats_enriched()
    
    for _, season_player_stats in player_stats.groupby('Season'):
        upload_data(season_player_stats, "Player_Stats_Enriched", overwrite=True, update_if_identical=True)
    
    return player_stats.to_json(orient='records')

@app.route("/preprocess/matchstats", methods=["GET", "POST"])
def create_match_stats(ID = None):
    
    match_stats = create_match_stats_enriched()
    
    # upload_data(Dataset_Name="Match_Stats_Enriched", Dataset=match_stats, overwrite=True, update_if_identical=True)
    
    return match_stats.to_json(orient='records')
    
    # return [True]

@app.route("/model/outcome/predict", methods=["GET", "POST"])
def predict_outcome(ID = None):
    
    model = load_outcome_model()
    model_features = model.xgb_model.get_booster().feature_names
        
    data = load_data(Dataset_Name="Match_Stats_Enriched", ID = request.json['ID'])      
    
    data = get_outcome_prediction(data, model, model_features)
    
    upload_data(Dataset_Name="CG_Match_Outcome", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')


@app.route("/model/margin/predict", methods=["GET", "POST"])
def predict_margin(ID = None):
    
    model = load_margin_model()
    model_features = model.xgb_model.get_booster().feature_names
        
    data = load_data(Dataset_Name="Match_Stats_Enriched", ID = request.json['ID'])      

    data = get_margin_prediction(data, model, model_features)
    
    upload_data(Dataset_Name="CG_Match_Margin", Dataset=data, overwrite=True, update_if_identical=True)
    
    return data.to_json(orient='records')

@app.route("/model/createtipping", methods=["GET", "POST"])
def create_tipping(ID = None):
    
    # Call predict match outcome and margin
    outcome_data = load_data(Dataset_Name="CG_Match_Outcome", ID = request.json['ID'])
    outcome_data = outcome_data[['Match_ID', 'Predicted_Team']]
    margin_data = load_data(Dataset_Name="CG_Match_Margin", ID = request.json['ID'])
    margin_data = margin_data[['Match_ID', 'Predicted_Margin']]
    
    tipping_data = outcome_data.merge(margin_data, how = "inner", on = "Match_ID")
    tipping_data['Predicted_Margin'] = tipping_data['Predicted_Margin'].astype(int)
    
    upload_data(Dataset_Name="CG_Tipping", Dataset=tipping_data, overwrite=True, update_if_identical=True)
    
    return tipping_data.to_json(orient='records')


@app.route("/model/tipping", methods=["GET", "POST"])
def apply_tipping(ID = None):
    
    data = load_data(Dataset_Name="CG_Tipping", ID = request.json['ID'])
    submit_tips(data)
    
    return get_current_tips(ID = request.json['ID'])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
    
