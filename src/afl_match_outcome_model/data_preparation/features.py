from afl_match_outcome_model.data_preparation.preprocessing import merge_venue_info, merge_home_away_venue
from afl_match_outcome_model.data_preparation.match_id_utils import get_home_team_from_match_id, get_away_team_from_match_id
from afl_match_outcome_model.data_preparation.preprocessing import create_home_flag
from afl_match_outcome_model.data_preparation.aggregation import aggregate_player_stats_to_team_stats
from afl_match_outcome_model.data_preparation.preprocessing import merge_match_summary_team_stats
from afl_match_outcome_model.data_preparation.elo import create_elo_ratings_home_away
from afl_match_outcome_model.data_preparation.feature_engineering import create_score_features, create_margin_features, create_win_features
from afl_match_outcome_model.data_preparation.rolling import create_team_rolling_features
from afl_match_outcome_model.data_preparation.feature_engineering import create_distance_travelled_feature, create_home_away_diff_feature

def create_team_stats(player_stats):
    
    player_stats['Home_Team'] = player_stats['Match_ID'].apply(get_home_team_from_match_id)
    player_stats['Away_Team'] = player_stats['Match_ID'].apply(get_away_team_from_match_id)
    player_stats['Home'] = create_home_flag(player_stats)

    return aggregate_player_stats_to_team_stats(
        player_stats, ['Player_Rating_Points']
    )

def create_features(matches, player_stats, venue_info, home_info, away_info):

    team_stats = create_team_stats(player_stats)
    
    rolling_feature_list = ['Win', 'Margin', 'Score', 'Goals', 'Player_Rating_Points', 'ELO']
    feature_diff_list = [f"{feature}_For_ewm5" for feature in rolling_feature_list] + ['Distance_Travelled']

    matches = merge_venue_info(matches, venue_info)
    matches = merge_home_away_venue(matches, home_info, away_info, venue_info)
    matches = merge_match_summary_team_stats(matches, team_stats)
    matches = create_elo_ratings_home_away(matches, 32)
    matches = create_score_features(matches)
    matches = create_margin_features(matches)
    matches = create_win_features(matches)
    matches = create_team_rolling_features(matches, rolling_feature_list, 5)
    matches = create_distance_travelled_feature(matches)
    matches = create_home_away_diff_feature(matches, feature_diff_list)

    return matches

