import pandas as pd

def calculate_elo_probability(home_team_elo, away_team_elo):
    """Calculates the probability of the home team winning a match based on their ELO ratings.

    Args:
        home_team_elo (float): The ELO rating of the home team.
        away_team_elo (float): The ELO rating of the away team.

    Returns:
        float: The probability of the home team winning the match.
    """
    
    return 1 / (1 + 10 ** ((away_team_elo - home_team_elo) / 400))

def calculate_elo(home_team_elo, away_team_elo, margin, k_factor):
    """Calculates the updated ELO ratings for the home and away teams based on the match outcome.

    Args:
        home_team_elo (float): The ELO rating of the home team.
        away_team_elo (float): The ELO rating of the away team.
        margin (int): The margin of victory in the match.
        k_factor (int): The size of the adjustment factor.

    Returns:
        tuple: A tuple containing the updated ELO ratings for the home and away teams.
            - new_home_team_elo (float): The updated ELO rating of the home team.
            - new_away_team_elo (float): The updated ELO rating of the away team.
    """
    
    prob_win_home = calculate_elo_probability(home_team_elo, away_team_elo)
    score_diff = 1 if margin > 0 else 0.5 if margin == 0 else 0
    new_home_team_elo = home_team_elo + k_factor * (score_diff - prob_win_home)
    new_away_team_elo = away_team_elo + k_factor * ((1 - score_diff) - (1 - prob_win_home))
    return new_home_team_elo, new_away_team_elo


def calculate_elo_for_row(row, elo_dict, k_factor):
    """Calculates the ELO ratings and probabilities for a single match.

    Args:
        row (Series): A pandas Series representing a single match.
        elo_dict (dict): A dictionary containing the ELO ratings for each team.
        k_factor (int): The size of the adjustment factor.

    Returns:
        tuple: A tuple containing the game_id, a list of home_team_elo and away_team_elo, and a list of prob_win_home and prob_win_away.
            - game_id (any): The identifier of the match.
            - home_team_elo (float): The ELO rating of the home team.
            - away_team_elo (float): The ELO rating of the away team.
            - prob_win_home (float): The probability of the home team winning the match.
            - prob_win_away (float): The probability of the away team winning the match.
    """
    
    game_id, home_team, away_team, margin = row['Match_ID'], row['Home_Team'], row['Away_Team'], row['Margin']
    home_team_elo, away_team_elo = elo_dict[home_team], elo_dict[away_team]
    prob_win_home = calculate_elo_probability(home_team_elo, away_team_elo)
    elo_dict[home_team], elo_dict[away_team] = calculate_elo(home_team_elo, away_team_elo, margin, k_factor)
    return game_id, [home_team_elo, away_team_elo], [prob_win_home, 1 - prob_win_home]

def calculate_elo_ratings(data, k_factor):
    """Calculates ELO ratings and probabilities for each team based on match data.

    Args:
        data (DataFrame): Input dataframe with match details.
        k_factor (int): The size of the adjustment factor.

    Returns:
        tuple: A tuple containing three dictionaries: elos, elo_probs, and elo_dict.
            - elos (dict): ELO ratings for each team before each match, with game_id as the key and a list of home_team_elo and away_team_elo as the value.
            - elo_probs (dict): Probabilities of each team winning each match, with game_id as the key and a list of prob_win_home and prob_win_away as the value.
            - elo_dict (dict): Final ELO ratings for each team at the end of the data, with team name as the key and the ELO rating as the value.
    """
    
    elo_dict = {team: 1500 for team in data['Home_Team'].unique()}
    elos, elo_probs = {}, {}

    for _, row in data.iterrows():
        game_id, elos_game, elo_probs_game = calculate_elo_for_row(row, elo_dict, k_factor)
        elos[game_id] = elos_game
        elo_probs[game_id] = elo_probs_game

    return elos, elo_probs, elo_dict


def convert_elo_dict_to_dataframe(elos, elo_probs):
    """Converts a dictionary of ELO ratings and ELO probabilities to dataframes for merging.

    Args:
        elos (dict): Dictionary of ELO ratings for each match.
        elo_probs (dict): Dictionary of ELO probabilities for each match.

    Returns:
        tuple: A tuple containing two dataframes: elo_df and elo_probs_df.
            - elo_df (DataFrame): Dataframe with ELO ratings for each match, including columns 'Match_ID', 'Home_ELO', 'Away_ELO', 'ELO_diff', and 'ELO_abs_diff'.
            - elo_probs_df (DataFrame): Dataframe with ELO probabilities for each match, including columns 'Match_ID', 'Home_ELO_probs', 'Away_ELO_probs', 'ELO_probs_diff', and 'ELO_probs_abs_diff'.
    """

    elo_df = pd.DataFrame.from_dict(elos, orient='index', columns=['Home_ELO', 'Away_ELO']).rename_axis('Match_ID').reset_index()
    elo_probs_df = pd.DataFrame.from_dict(elo_probs, orient='index', columns=['Home_ELO_probs', 'Away_ELO_probs']).rename_axis('Match_ID').reset_index()

    return elo_df, elo_probs_df


def merge_elo_ratings(X, elos, elo_probs):
    """ Merge ELO factors back onto original dataframe by Match ID

    Args:
        X (Dataframe): Original ELO calculation dataframe with Match ID
        elos_df (Dataframe): ELO ratings for each team before each match by Match_ID
        elo_probs_df (Dataframe): Probabilities of each team winning each match by Match_ID

    Returns:
        Dataframe : Input data with ELO columns merged on.
    """
    
    elo_df, elo_probs_df = convert_elo_dict_to_dataframe(elos, elo_probs)
    
    X = pd.merge(X, elo_df, how = 'left', on = 'Match_ID')
    X = pd.merge(X, elo_probs_df, how = 'left', on = 'Match_ID')
    
    return X


def create_elo_ratings_home_away(data, k_factor=32):
    """Calculate ELO ratings and probabilities for each team based on home and away matches.

    Args:
        data (DataFrame): Input dataframe with match details.
        k_factor (int, optional): The size of the adjustment factor. Defaults to 32.

    Returns:
        DataFrame: Merged dataframe with ELO ratings and probabilities for each team.
    """
    elos, elo_probs, _ = calculate_elo_ratings(data, k_factor)
    return merge_elo_ratings(data, elos, elo_probs)
