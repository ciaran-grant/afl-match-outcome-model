import joblib
from flask import Flask, request
from AFLPy.AFLData_Client import load_data, upload_data

app = Flask(__name__)

def load_model():
    
    model_file_path = "model_outputs/match_outcome_xgb_v8.joblib"
    return joblib.load(model_file_path)

@app.route("/model/outcome/predict", methods=["GET", "POST"])
def predict(ID = None):
    
    model = load_model()
    model_features = model.xgb_model.get_booster().feature_names
        
    data = load_data(Dataset_Name="Match_Stats_Enriched", ID = request.json['ID'])      
    
    data['Home_Win_Prob'] = model.predict_proba(data[model_features])[:, 1]
    data['Away_Win_Prob'] = 1 - data['Home_Win_Prob']
    
    data = data[['Match_ID', 'Home_Win_Prob', 'Away_Win_Prob'] + model_features]
    
    upload_data(Dataset_Name="CG_Match_Outcome", Dataset=data, overwrite=True)
    
    return data.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
    
