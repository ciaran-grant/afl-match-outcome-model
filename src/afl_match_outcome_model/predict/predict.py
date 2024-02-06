from AFLPy.AFLData_Client import load_data

def predict_outcome(match_id, model):
    
    # Get features
    model_features = model.xgb_model.get_booster().feature_names    
    
    # Load features
    match_stats = load_data(Dataset_Name='CG_Match_Outcome_Features', ID = match_id)

    return model.predict_proba(match_stats[model_features])