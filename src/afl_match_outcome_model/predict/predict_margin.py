import joblib

def load_margin_model():
    
    model_file_path = "model_outputs/match_margin_xgb_v10.joblib"
    
    return joblib.load(model_file_path)

def load_margin_preprocessor():
    
    preproc_file_path = "model_outputs/match_margin_pipeline_v10.joblib"
    
    return joblib.load(preproc_file_path)