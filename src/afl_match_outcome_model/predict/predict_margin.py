import joblib

def load_margin_model():
    
    model_file_path = "model_outputs/match_margin_xgb_v9.joblib"
    
    return joblib.load(model_file_path)

def get_margin_prediction(data, model, model_features):
    
    data['Predicted_Margin'] = model.predict(data[model_features]).astype(int)

    data = data[['Match_ID', 'Predicted_Margin'] + model_features]
    
    return data