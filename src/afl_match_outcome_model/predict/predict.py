from afl_match_outcome_model.data_preparation.data_loader import load_match_outcome_features

def predict_outcome(match_id, model):
    
    # Get features
    model_features = model.xgb_model.get_booster().feature_names    
    
    # Load features
    match_stats = load_match_outcome_features(dataset_name='CG_Match_Outcome_Features', match_id=match_id)

    return model.predict_proba(match_stats[model_features])