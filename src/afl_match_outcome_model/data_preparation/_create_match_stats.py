from AFLPy.AFLData_Client import load_data
import pandas as pd
from typing import Dict, List

from afl_match_outcome_model.data_preparation.preprocessing import merge_venue_info, merge_home_away_venue
from afl_match_outcome_model.data_preparation.match_id_utils import get_home_team_from_match_id, get_away_team_from_match_id
from afl_match_outcome_model.data_preparation.preprocessing import create_home_flag
from afl_match_outcome_model.data_preparation.utils import get_numeric_columns_list
from afl_match_outcome_model.data_preparation.aggregation import aggregate_player_stats_to_team_stats
from afl_match_outcome_model.data_preparation.preprocessing import merge_match_summary_team_stats
from afl_match_outcome_model.data_preparation.elo import create_elo_ratings_home_away
from afl_match_outcome_model.data_preparation.feature_engineering import create_score_features, create_margin_features, create_win_features, create_shot_features, create_expected_features
from afl_match_outcome_model.data_preparation.rolling import create_team_rolling_features
from afl_match_outcome_model.data_preparation.feature_engineering import create_distance_travelled_feature, create_home_away_diff_feature
from afl_match_outcome_model.data_preparation.squad import get_squad_stats

def load_match_data() -> Dict[str, pd.DataFrame]:
    """
    Loads the match data and related datasets.

    Returns:
        A dictionary containing the loaded datasets:
        - 'matches': The AFL API matches data.
        - 'fryzigg_match_summary': The Fryzigg match summary data.
        - 'footywire_match_summary': The Footywire match summary data.
        - 'afltables_match_summary': The AFLTables match summary data.
        - 'venue': The venues data.
        - 'home_info': The team information data with renamed columns for the home team.
        - 'away_info': The team information data with renamed columns for the away team.

    Examples:
        data = load_match_data()
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
        'home_info': load_data(Dataset_Name='Team_Info').rename(columns={'Team': 'Home_Team', 'Home_Ground_1': 'Home_Team_Venue'}),
        'away_info': load_data(Dataset_Name='Team_Info').rename(columns={'Team': 'Away_Team', 'Home_Ground_1': 'Away_Team_Venue'}),
    }

def calculate_missing_columns_and_merge(
    base_df: pd.DataFrame, 
    summary_df: pd.DataFrame, 
    id_columns: List[str]):
    """
    Calculates the missing columns between the base DataFrame and the summary DataFrame and merges them.

    Args:
        base_df: The base DataFrame.
        summary_df: The summary DataFrame.
        id_columns: The columns used for merging.

    Returns:
        A merged DataFrame containing the base DataFrame and the missing columns from the summary DataFrame.

    Raises:
        None

    Examples:
        merged_data = calculate_missing_columns_and_merge(base_df, summary_df, id_columns)
    """
    missing_columns = list(set(summary_df.columns) - set(base_df.columns))
    return base_df.merge(summary_df[id_columns + missing_columns], how="left", on=id_columns)

def combine_match_summary(match_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines the match data and various match summaries into a single DataFrame.

    Args:
        match_data_dict: A dictionary containing the match data and summaries.

    Returns:
        A DataFrame containing the combined match data and summaries.

    Raises:
        None

    Examples:
        combined_data = combine_match_summary(match_data_dict)
    """
    id_columns = ['Match_ID']
    matches = match_data_dict['matches']
    summaries = [match_data_dict[key] for key in ['fryzigg_match_summary', 'footywire_match_summary', 'afltables_match_summary']]
    venues = match_data_dict['venue']
    home_info, away_info = match_data_dict['home_info'], match_data_dict['away_info']

    match_stats_enriched = matches
    for summary in summaries:
        match_stats_enriched = calculate_missing_columns_and_merge(match_stats_enriched, summary, id_columns)

    match_stats_enriched = merge_venue_info(match_stats_enriched, venues)
    match_stats_enriched = merge_home_away_venue(match_stats_enriched, home_info, away_info, venues)

    match_stats_enriched[['Year', 'Round']] = match_stats_enriched['Match_ID'].str.split("_", expand=True)[[1, 2]]

    return match_stats_enriched

def create_team_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Creates team statistics from player statistics.

    Args:
        player_stats: A DataFrame containing player statistics.

    Returns:
        A DataFrame containing team statistics.

    Raises:
        None

    Examples:
        team_stats = create_team_stats(player_stats_df)
    """
    player_stats['Home_Team'] = player_stats['Match_ID'].apply(get_home_team_from_match_id)
    player_stats['Away_Team'] = player_stats['Match_ID'].apply(get_away_team_from_match_id)
    player_stats['Home'] = create_home_flag(player_stats)

    return aggregate_player_stats_to_team_stats(
        player_stats, get_numeric_columns_list(player_stats)
    )

def load_player_stats_enriched() -> pd.DataFrame:
    """
    Loads the enriched player statistics data.

    Returns:
        A DataFrame containing the enriched player statistics.

    Examples:
        player_stats = load_player_stats_enriched()
    """
    player_stats_list = [load_data(Dataset_Name='Player_Stats_Enriched', ID=f"AFL_{season}") for season in list(range(2020, 2024))]
    player_stats = pd.concat(player_stats_list, axis=0)
    player_stats = player_stats.sort_values(by="Match_ID")

    return player_stats

def get_match_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves the match statistics from the given matches DataFrame.

    Args:
        matches: A DataFrame containing the match data.

    Returns:
        A DataFrame containing the match statistics.

    Raises:
        None

    Examples:
        match_stats = get_match_stats(matches_df)
    """
    player_stats = load_player_stats_enriched()

    team_stats = create_team_stats(player_stats)
    matches = merge_match_summary_team_stats(matches, team_stats)

    matches = get_squad_stats(matches, player_stats)

    matches = matches.sort_values(by='Match_ID')

    return matches

def create_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    
    rolling_feature_list = [
        'Disposals',
        'Dream_Team_Points',
        'ELO',
        'Goals',
        'Handballs',
        'Inside_50s',
        'Kicks',
        'Margin',
        'Score',
        'Scoring_Shots',
        'Shots_At_Goal',
        'Win',
        'AFL_Fantasy_Points',
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
        'Scoring_Shots_Differential'
        ]
    
    for window in [5, 10]:
        matches = create_team_rolling_features(matches, rolling_feature_list, window)
        
        feature_diff_list = [f"{feature}_For_ewm{window}" for feature in rolling_feature_list] + [f"{feature}_Against_ewm{window}" for feature in rolling_feature_list]
        matches = create_home_away_diff_feature(matches, feature_diff_list)
        
    return matches

def create_match_stats_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Creates match statistics features from the given matches DataFrame.

    Args:
        matches: A DataFrame containing the match data.

    Returns:
        A DataFrame containing the match statistics features.

    Examples:
        match_stats_features = create_match_stats_features(matches_df)
    """
    return (matches
            .pipe(create_elo_ratings_home_away, 32)
            .pipe(create_score_features)
            .pipe(create_margin_features)
            .pipe(create_win_features)
            .pipe(create_shot_features)
            .pipe(create_expected_features)
            .pipe(create_distance_travelled_feature)
            .pipe(create_home_away_diff_feature, ['Distance_Travelled'])
            .pipe(create_rolling_features)
            .sort_values(by='Match_ID'))

