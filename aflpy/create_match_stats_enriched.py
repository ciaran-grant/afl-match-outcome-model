from AFLPy.AFLData_Client import load_data, upload_data

from afl_match_outcome_model.data_preparation.preprocessing import merge_venue_info, merge_home_away_venue
from afl_match_outcome_model.data_preparation.match_id_utils import get_home_team_from_match_id, get_away_team_from_match_id
from afl_match_outcome_model.data_preparation.preprocessing import create_home_flag
from afl_match_outcome_model.data_preparation.utils import get_numeric_columns_list
from afl_match_outcome_model.data_preparation.aggregation import aggregate_player_stats_to_team_stats
from afl_match_outcome_model.data_preparation.preprocessing import merge_match_summary_team_stats
from afl_match_outcome_model.data_preparation.elo import create_elo_ratings_home_away
from afl_match_outcome_model.data_preparation.feature_engineering import create_score_features, create_margin_features, create_win_features, create_expected_features, create_shot_features
from afl_match_outcome_model.data_preparation.rolling import create_team_rolling_features
from afl_match_outcome_model.data_preparation.feature_engineering import create_distance_travelled_feature, create_home_away_diff_feature


def load_match_data():
    """
    Loads match data from various datasets.

    Returns:
        dict: A dictionary containing the following keys:
            - 'matches': Match data from AFL_API_Matches dataset.
            - 'fryzigg_match_summary': Match summary data from Fryzigg_Match_Summary dataset.
            - 'footywire_match_summary': Match summary data from Footywire_Match_Summary dataset.
            - 'afltables_match_summary': Match summary data from AFLTables_Match_Summary dataset.
            - 'venue': Venue data from Venues dataset.
            - 'home_info': Team information data from Team_Info dataset, with columns renamed for home team.
            - 'away_info': Team information data from Team_Info dataset, with columns renamed for away team.
    """

    return {
        'matches': load_data(
            Dataset_Name="AFL_API_Matches", ID="AFL"
        ),
        'fryzigg_match_summary': load_data(
            Dataset_Name="Fryzigg_Match_Summary", ID="AFL"
        ),
        'footywire_match_summary': load_data(
            Dataset_Name="Footywire_Match_Summary", ID="AFL"
        ),
        'afltables_match_summary': load_data(
            Dataset_Name="AFLTables_Match_Summary", ID="AFL"
        ),
        'venue': load_data(
            Dataset_Name='Venues'
        ),
        'home_info':load_data(Dataset_Name='Team_Info').rename(columns = {'Team':'Home_Team', 'Home_Ground_1':'Home_Team_Venue'}),
        'away_info':load_data(Dataset_Name='Team_Info').rename(columns = {'Team':'Away_Team', 'Home_Ground_1':'Away_Team_Venue'}),
        
    }

def combine_match_summary(match_data_dict):
    """
    Combines match summary data from different datasets into a single DataFrame.

    Args:
        match_data_dict (dict): A dictionary containing the following keys:
            - 'matches': Match data from AFL_API_Matches dataset.
            - 'fryzigg_match_summary': Match summary data from Fryzigg_Match_Summary dataset.
            - 'footywire_match_summary': Match summary data from Footywire_Match_Summary dataset.
            - 'afltables_match_summary': Match summary data from AFLTables_Match_Summary dataset.
            - 'venue': Venue data from Venues dataset.
            - 'home_info': Team information data from Team_Info dataset, with columns renamed for home team.
            - 'away_info': Team information data from Team_Info dataset, with columns renamed for away team.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined match summary data enriched with venue information,
        home and away team information, year, and round.

    """

    
    id_columns = ['Match_ID']
    matches = match_data_dict['matches']
    fryzigg_match_summary = match_data_dict['fryzigg_match_summary']
    footywire_match_summary = match_data_dict['footywire_match_summary']
    afltables_match_summary = match_data_dict['afltables_match_summary']
    venues = match_data_dict['venue']
    home_info, away_info = match_data_dict['home_info'], match_data_dict['away_info']
    
    missing_fryzigg = list(set(list(fryzigg_match_summary)) - set(list(matches)))
    match_stats_enriched = matches.merge(fryzigg_match_summary[id_columns + missing_fryzigg], how = "left", on = id_columns)

    missing_footywire = list(set(list(footywire_match_summary)) - set(list(match_stats_enriched)))
    match_stats_enriched = match_stats_enriched.merge(footywire_match_summary[id_columns + missing_footywire], how = "left", on = id_columns)

    missing_afltable = list(set(list(afltables_match_summary)) - set(list(match_stats_enriched)))
    match_stats_enriched = match_stats_enriched.merge(afltables_match_summary[id_columns + missing_afltable], how = "left", on = id_columns)

    match_stats_enriched = merge_venue_info(match_stats_enriched, venues)
    match_stats_enriched = merge_home_away_venue(match_stats_enriched, home_info, away_info, venues)
    
    match_stats_enriched['Year'] = match_stats_enriched['Match_ID'].apply(lambda x: x.split("_")[1])
    match_stats_enriched['Round'] = match_stats_enriched['Match_ID'].apply(lambda x: x.split("_")[2])
    
    return match_stats_enriched

def create_team_stats(player_stats):
    """
    Creates team-level statistics from player-level statistics.

    Args:
        player_stats (pandas.DataFrame): DataFrame containing player-level statistics.

    Returns:
        pandas.DataFrame: DataFrame containing aggregated team-level statistics.

    """

    
    player_stats['Home_Team'] = player_stats['Match_ID'].apply(get_home_team_from_match_id)
    player_stats['Away_Team'] = player_stats['Match_ID'].apply(get_away_team_from_match_id)
    player_stats['Home'] = create_home_flag(player_stats)

    return aggregate_player_stats_to_team_stats(
        player_stats, get_numeric_columns_list(player_stats)
    )
    
def get_match_stats(matches):
    """
    Retrieves match statistics by merging match data with player and team statistics.

    Args:
        matches (pandas.DataFrame): DataFrame containing match data.

    Returns:
        pandas.DataFrame: DataFrame containing match statistics enriched with player and team statistics.

    """

    
    player_stats = load_data(Dataset_Name="Player_Stats_Enriched", ID = "AFL")
    
    team_stats = create_team_stats(player_stats)
    matches = merge_match_summary_team_stats(matches, team_stats)
    
    matches = matches.sort_values(by = 'Match_ID')
    
    return matches

def create_match_stats_features(matches):
    """
    Creates additional match statistics features based on the given matches DataFrame.

    Args:
        matches (pandas.DataFrame): DataFrame containing match data.

    Returns:
        pandas.DataFrame: DataFrame containing the matches data with additional match statistics features.

    """


    matches = create_elo_ratings_home_away(matches, 32)
    matches = create_score_features(matches)
    matches = create_margin_features(matches)
    matches = create_win_features(matches)
    matches = create_shot_features(matches)
    matches = create_expected_features(matches)
    matches = create_distance_travelled_feature(matches)
    matches = create_home_away_diff_feature(matches, ['Distance_Travelled'])

    rolling_feature_list = [
        'Behinds',
        'Disposals',
        'Dream_Team_Points',
        'Effective_Disposals',
        'Effective_Kicks',
        'ELO',
        'Goals',
        'Handballs',
        'Inside_50s',
        'Kicks',
        'Margin',
        'Marks',
        'Metres_Gained',
        'Rating_Points',
        'Score',
        'Scoring_Shots',
        'Shots_At_Goal',
        'Win',
        'AFL_Fantasy_Points',
        'Super_Coach_Points',
        'Player_Rating_Points',
        'Brownlow_Votes',
        'Coaches_Votes',
        'xScore',
        'xT_created', 
        'xT_denied',
        'vaep_value', 
        'offensive_value', 
        'defensive_value', 
        'exp_vaep_value', 
        'exp_offensive_value', 
        'exp_defensive_value',
        'xMargin',
        'xVAEP_Margin',
        # 'Scoring_Shot_Differential'
        ]
    for window in [5, 10]:
        matches = create_team_rolling_features(matches, rolling_feature_list, window)
        feature_diff_list = [f"{feature}_For_ewm{window}" for feature in rolling_feature_list] + [f"{feature}_Against_ewm{window}" for feature in rolling_feature_list]
        matches = create_home_away_diff_feature(matches, feature_diff_list)
        
    matches = matches.sort_values(by = 'Match_ID')

    return matches

def update_match_stats_enriched():
    match_data_dict = load_match_data()
    match_stats_enriched = combine_match_summary(match_data_dict)
    
    match_stats_enriched = get_match_stats(match_stats_enriched)
    match_stats_enriched = create_match_stats_features(match_stats_enriched)
    match_stats_enriched = match_stats_enriched.sort_values(by = 'Match_ID')
    
    upload_data(match_stats_enriched, "Match_Stats_Enriched", overwrite=True, update_if_identical=True)
    
if __name__ == "__main__":
    update_match_stats_enriched()