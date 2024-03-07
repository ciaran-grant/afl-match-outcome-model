import pandas as pd
from afl_match_outcome_model.data_preparation._create_player_stats import load_player_data, load_expected_player_data, combine_player_stats, create_player_stats_features

def create_player_stats_enriched() -> pd.DataFrame:
    """
    Updates the player statistics enriched data by combining player data from different datasets,
    creating additional player statistics features, and uploading the updated data.

    Returns:
        None

    """

    player_data_dict = load_player_data()
    player_expected_data = load_expected_player_data()
    player_stats_enriched = combine_player_stats(player_data_dict, player_expected_data)
    player_stats_enriched_features = create_player_stats_features(player_stats_enriched)
    
    player_stats_enriched_features['Year'] = player_stats_enriched_features['Match_ID'].apply(lambda x: x.split("_")[1])
    player_stats_enriched_features['Season'] = player_stats_enriched_features['Match_ID'].apply(lambda x: x.split("_")[1])
    
    return player_stats_enriched_features
