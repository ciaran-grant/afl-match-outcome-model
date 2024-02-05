from afl_match_outcome_model.data_preparation.data_loader import load_matches, load_player_stats, load_venues, load_team_info
from afl_match_outcome_model.data_preparation.features import create_features
from AFLPy.AFLData_Client import upload_data

def update_match_outcome_features():

    # Load data (alternative to loading raw data and then creating features would be to upload data with features already created)
    matches = load_matches(dataset_name = 'AFLTables_Match_Summary')
    player_stats = load_player_stats(dataset_name = 'Fryzigg_Player_Stats')
    venue_info = load_venues(dataset_name = 'Venues')
    home_info, away_info = load_team_info(dataset_name = 'Team_Info')

    # Create Features
    match_stats = create_features(matches, player_stats, venue_info, home_info, away_info)

    # Load Features to AFLData
    upload_data(Dataset=match_stats, Dataset_Name="CG_Match_Outcome_Features", overwrite=True, update_if_identical=False, tags=[])
    
if __name__ == "__main__":
    update_match_outcome_features()