import joblib
import numpy as np

def load_outcome_model():
    
    model_file_path = "model_outputs/match_outcome_xgb_v10.joblib"
    
    return joblib.load(model_file_path)

def get_outcome_prediction(data, model, model_features):
    
    data['Home_Win_Prob'] = model.predict_proba(data[model_features])[:, 1]
    data['Away_Win_Prob'] = 1 - data['Home_Win_Prob']
    
    data['Predicted_Team'] = np.where(data['Home_Win_Prob'] > 0.5, data['Home_Team'], data['Away_Team'])
    
    data = data[['Match_ID', 'Home_Win_Prob', 'Away_Win_Prob', 'Predicted_Team'] + model_features]
    
    return data

def load_outcome_preprocessor():
    
    preproc_file_path = "model_outputs/match_outcome_pipeline_v10.joblib"
    
    return joblib.load(preproc_file_path)
