import joblib
import numpy as np

def load_margin_model():
    
    model_file_path = "model_outputs/match_margin_xgb_v9.joblib"
    
    return joblib.load(model_file_path)

def get_margin_prediction(data, model, model_features):
    
    data['Home_Margin_Prediction'] = model.predict(data[model_features])
    data['Predicted_Team'] = np.where(data['Home_Margin_Prediction'] > 0, data['Home_Team'], data['Away_Team'])

    data = data[['Match_ID', 'Home_Margin_Prediction', 'Predicted_Team'] + model_features]
    
    return data