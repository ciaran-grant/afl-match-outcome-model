import joblib
from afl_match_outcome_model.data_preparation.data_loader import load_matches, load_player_stats, load_venues, load_team_info
from afl_match_outcome_model.data_preparation.features import create_features
from afl_match_outcome_model.data_preparation.preprocessing import get_match

# Specify Match_ID to predict
match_id = "AFL_2023_F4_Collingwood_Brisbane"

def predict_outcome(match_id):
    
    # Specify model to load
    model_file_path = "model_outputs/match_outcome_xgb.joblib"
    super_xgb = joblib.load(model_file_path)
    model_features = super_xgb.xgb_model.get_booster().feature_names    
    
    # Load data (alternative to loading raw data and then creating features would be to upload data with features already created)
    matches = load_matches(dataset_name = 'AFLTables_Match_Summary')
    player_stats = load_player_stats(dataset_name = 'Fryzigg_Player_Stats')
    venue_info = load_venues(dataset_name = 'Venues')
    home_info, away_info = load_team_info(dataset_name = 'Team_Info')

    # Create Features
    match_stats = create_features(matches, player_stats, venue_info, home_info, away_info)

    # Get correct row
    data = get_match(match_stats, match_id)

    return super_xgb.predict_proba(data[model_features])