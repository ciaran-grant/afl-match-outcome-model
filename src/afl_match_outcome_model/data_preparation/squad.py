import pandas as pd
from AFLPy.AFLData_Client import load_data
from afl_match_outcome_model.data_preparation.match_id_utils import get_home_team_from_match_id, get_away_team_from_match_id

def get_squad_list_from_match_id(ID):
    """
    Retrieves the squad list for the home and away teams from a given match ID.

    Args:
        ID: The match ID.

    Returns:
        A tuple containing the home squad list and the away squad list.

    Examples:
        home_squad, away_squad = get_squad_list_from_match_id("AFL_2023_F4_Collingwood_Brisbane")
    """
    team_positions = load_data(Dataset_Name="AFL_API_Team_Positions", ID=ID)
    home_team = get_home_team_from_match_id(match_id=ID)
    away_team = get_away_team_from_match_id(match_id=ID)

    home_squad = list(team_positions[team_positions['Team'] == home_team]['Player'])
    away_squad = list(team_positions[team_positions['Team'] == away_team]['Player'])

    return home_squad, away_squad

def aggregate_squad_team_stats(new_player_stats, match_id, home_away="Home"):
    """
    Aggregates the squad team statistics by summing the player statistics.

    Args:
        new_player_stats: A DataFrame containing the player statistics.
        match_id: The match ID.
        home_away: The team type, either "Home" or "Away" (default: "Home").

    Returns:
        A DataFrame containing the aggregated squad team statistics.

    Raises:
        None

    Examples:
        new_squad_team_stats = aggregate_squad_team_stats(new_player_stats, "AFL_2023_F4_Collingwood_Brisbane", home_away="Away")
    """
    new_squad_team_stats = pd.DataFrame(new_player_stats.sum(numeric_only=True)).T
    new_squad_team_stats.columns = [f'{home_away}_{x}_Squad_sum' for x in list(new_squad_team_stats)]
    new_squad_team_stats.insert(0, 'Match_ID', match_id)

    return new_squad_team_stats

def get_latest_squad_player_stats(match_id, player_stats, numeric_stats):
    """
    Retrieves the latest player statistics for the home and away squads of a given match ID.

    Args:
        match_id: The match ID.
        player_stats: A DataFrame containing the player statistics.
        numeric_stats: A list of numeric statistics to retrieve.

    Returns:
        A tuple containing the latest player statistics for the home squad and the away squad.

    Raises:
        None

    Examples:
        latest_home_stats, latest_away_stats = get_latest_squad_player_stats("AFL_2023_F4_Collingwood_Brisbane", player_stats_df, ['Goals', 'Player_Rating_Points'])
    """
    home_squad, away_squad = get_squad_list_from_match_id(ID=match_id)

    home_team = get_home_team_from_match_id(match_id=match_id)
    away_team = get_away_team_from_match_id(match_id=match_id)

    home_player_stats = player_stats[(player_stats['Team'] == home_team)][['Match_ID', 'Player'] + numeric_stats]
    away_player_stats = player_stats[(player_stats['Team'] == away_team)][['Match_ID', 'Player'] + numeric_stats]

    home_squad_player_stats = home_player_stats[home_player_stats['Player'].isin(home_squad)]
    away_squad_player_stats = away_player_stats[away_player_stats['Player'].isin(away_squad)]

    latest_home_stats = home_squad_player_stats.groupby('Player')[numeric_stats].last().reset_index()
    latest_away_stats = away_squad_player_stats.groupby('Player')[numeric_stats].last().reset_index()

    return latest_home_stats, latest_away_stats

def aggregate_player_stats_by_match_squad(match_id, player_stats, numeric_stats):
    """
    Aggregates the player statistics by match squad for a given match ID.

    Args:
        match_id: The match ID.
        player_stats: A DataFrame containing the player statistics.
        numeric_stats: A list of numeric statistics to aggregate.

    Returns:
        A DataFrame containing the aggregated squad team statistics.

    Examples:
        new_squad_team_stats = aggregate_player_stats_by_match_squad("AFL_2023_F4_Collingwood_Brisbane", player_stats_df, ['Goals', 'Player_Rating_Points'])
    """
    latest_home_squad_player_stats, latest_away_squad_player_stats = get_latest_squad_player_stats(match_id, player_stats, numeric_stats)

    new_home_squad_team_stats = aggregate_squad_team_stats(latest_home_squad_player_stats, match_id=match_id, home_away='Home')
    new_away_squad_team_stats = aggregate_squad_team_stats(latest_away_squad_player_stats, match_id=match_id, home_away='Away')

    new_squad_team_stats = new_home_squad_team_stats.merge(new_away_squad_team_stats, how="left", on="Match_ID")

    for col in numeric_stats:
        new_squad_team_stats[f'{col}_Squad_sum_diff'] = new_squad_team_stats[f'Home_{col}_Squad_sum'] - new_squad_team_stats[f'Away_{col}_Squad_sum']

    return new_squad_team_stats


def get_squad_stats(matches, player_stats):
    """
    Retrieves the squad statistics for each match in the given matches DataFrame.

    Args:
        matches: A DataFrame containing the match data.
        player_stats: A DataFrame containing the player statistics.

    Returns:
        A DataFrame containing the matches data merged with the aggregated squad player statistics.

    Examples:
        squad_stats = get_squad_stats(matches_df, player_stats_df)
    """
    squads = load_data(Dataset_Name='AFL_API_Team_Positions', ID="AFL")

    numeric_stats = [
        'Goals_ewm10',
        'Player_Rating_Points_ewm10',
        'xScore_ewm10',
        'xT_created_ewm10',
        'vaep_value_ewm10',
        'offensive_value_ewm10',
        'exp_vaep_value_ewm10',
        'exp_offensive_value_ewm10'
    ]

    match_id_history = sorted(list(squads['Match_ID'].unique()))
    match_id_history_post_2020 = [x for x in match_id_history if int(x.split("_")[1]) > 2020]

    match_squad_player_stats_list = [aggregate_player_stats_by_match_squad(match_id, player_stats, numeric_stats) for
                                     match_id in match_id_history_post_2020]

    match_squad_player_stats = pd.concat(match_squad_player_stats_list, axis=0)
    matches = matches.merge(match_squad_player_stats, how="left", on="Match_ID")

    return matches
