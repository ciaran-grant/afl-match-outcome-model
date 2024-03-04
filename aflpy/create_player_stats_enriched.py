from AFLPy.AFLData_Client import load_data, upload_data
import pandas as pd

player_columns = [
    'Goals', 
    'Behinds',
    'Disposals',
    'Kicks',
    'Handballs',
    'Shots_At_Goal',
    'Player_Rating_Points', 
    'xScore', 
    'xT_created', 
    'xT_denied',
    'vaep_value', 
    'offensive_value', 
    'defensive_value', 
    'exp_vaep_value', 
    'exp_offensive_value', 
    'exp_defensive_value']

def load_player_data():
    """
    Loads player data from various datasets.

    Returns:
        dict: A dictionary containing the following keys:
            - 'player_stats': Player statistics data from AFL_API_Player_Stats dataset.
            - 'fryzigg_player_stats': Player statistics data from Fryzigg_Player_Stats dataset.
            - 'footywire_player_stats': Player statistics data from Footywire_Player_Stats dataset.
            - 'afltables_player_stats': Player statistics data from AFLTables_Player_Stats dataset.
            - 'coaches_votes': Coaches votes data from AFLCA_Coaches_Votes dataset.

    """

    
    return {
        'player_stats': load_data(
            Dataset_Name="AFL_API_Player_Stats", ID="AFL"
        ),
        'fryzigg_player_stats': load_data(
            Dataset_Name="Fryzigg_Player_Stats", ID="AFL"
        ),
        'footywire_player_stats': load_data(
            Dataset_Name="Footywire_Player_Stats", ID="AFL"
        ),
        'afltables_player_stats': load_data(
            Dataset_Name="AFLTables_Player_Stats", ID="AFL"
        ),
        'coaches_votes': load_data(
            Dataset_Name='AFLCA_Coaches_Votes', ID="AFL"
        ),
    }
    
def load_expected_player_data():
    """
    Loads expected player data from a CSV file and performs data transformations.

    Returns:
        pandas.DataFrame: DataFrame containing the expected player data with selected columns.

    """

    
    expected_player_stats = pd.read_csv("/Users/ciaran/Documents/Projects/AFL/data/scored_player_stats_v2.csv")
    expected_player_stats['Match_ID'] = expected_player_stats['Match_ID'].apply(
        lambda x: "AFL_"+x.split("_")[0][:4]+"_"+x.split("_")[0][-2:]+"_"+x.split("_")[1]+"_"+x.split("_")[-1]
    )
    expected_player_stats['Match_ID'] = expected_player_stats['Match_ID'].apply(lambda x: x.replace("BrisbaneLions", "Brisbane"))
    expected_cols = ['xScore', 'xT_created', 'xT_denied', 'vaep_value', 'offensive_value', 'defensive_value', 'exp_vaep_value', 'exp_offensive_value', 'exp_defensive_value']
    
    return expected_player_stats[['Match_ID', 'Player'] + expected_cols]

def combine_player_stats(player_data_dict, expected_player_stats):
    """
    Combines player statistics from different datasets into a single DataFrame.

    Args:
        player_data_dict (dict): A dictionary containing the following keys:
            - 'player_stats': Player statistics data from AFL_API_Player_Stats dataset.
            - 'fryzigg_player_stats': Player statistics data from Fryzigg_Player_Stats dataset.
            - 'footywire_player_stats': Player statistics data from Footywire_Player_Stats dataset.
            - 'afltables_player_stats': Player statistics data from AFLTables_Player_Stats dataset.
            - 'coaches_votes': Coaches votes data from AFLCA_Coaches_Votes dataset.
        expected_player_stats (pandas.DataFrame): DataFrame containing expected player statistics.

    Returns:
        pandas.DataFrame: DataFrame containing the combined player statistics enriched with expected player statistics.

    """

    
    id_columns = ['Player', 'Match_ID']
    player_stats = player_data_dict['player_stats']
    fryzigg_player_stats = player_data_dict['fryzigg_player_stats']
    footywire_player_stats = player_data_dict['footywire_player_stats']
    afltables_player_stats = player_data_dict['afltables_player_stats']
    coaches_votes = player_data_dict['coaches_votes']
    
    missing_fryzigg_stats = list(set(list(fryzigg_player_stats)) - set(list(player_stats)))
    player_stats_enriched = player_stats.merge(fryzigg_player_stats[id_columns + missing_fryzigg_stats], how = "left", on = id_columns)

    missing_footywire_stats = list(set(list(footywire_player_stats)) - set(list(player_stats_enriched)))
    player_stats_enriched = player_stats_enriched.merge(footywire_player_stats[id_columns + missing_footywire_stats], how = "left", on = id_columns)

    missing_afltable_stats = list(set(list(afltables_player_stats)) - set(list(player_stats_enriched)))
    player_stats_enriched = player_stats_enriched.merge(afltables_player_stats[id_columns + missing_afltable_stats], how = "left", on = id_columns)

    player_stats_enriched = player_stats_enriched.merge(coaches_votes[id_columns + ['Coaches_Votes']], how = "left", on = id_columns)
    player_stats_enriched['Coaches_Votes'] = player_stats_enriched['Coaches_Votes'].fillna(0)
    
    player_stats_enriched = player_stats_enriched.merge(expected_player_stats, how = "left", on = id_columns)

    return player_stats_enriched

def calculate_rolling_avg5(player_group, columns):
    """
    Calculates the rolling average with a window size of 5 for the specified columns within a player group.

    Args:
        player_group (pandas.DataFrame): DataFrame representing a group of player data.
        columns (list): List of column names to calculate the rolling average for.

    Returns:
        pandas.DataFrame: DataFrame containing the rolling average values for the specified columns, shifted by 1.

    """

    return player_group[columns].rolling(window=5).mean().shift(1)

def calculate_rolling_avg10(player_group, columns):
    """
    Calculates the rolling average with a window size of 10 for the specified columns within a player group.

    Args:
        player_group (pandas.DataFrame): DataFrame representing a group of player data.
        columns (list): List of column names to calculate the rolling average for.

    Returns:
        pandas.DataFrame: DataFrame containing the rolling average values for the specified columns, shifted by 1.

    """
    return player_group[columns].rolling(window=10).mean().shift(1)

def calculate_ewm5(player_group, columns):
    """
    Calculates the exponentially weighted moving average with a span of 5 for the specified columns within a player group.

    Args:
        player_group (pandas.DataFrame): DataFrame representing a group of player data.
        columns (list): List of column names to calculate the exponentially weighted moving average for.

    Returns:
        pandas.DataFrame: DataFrame containing the exponentially weighted moving average values for the specified columns, shifted by 1.

    """

    return player_group[columns].ewm(span=5, adjust = False).mean().shift(1)

def calculate_ewm10(player_group, columns):
    """
    Calculates the exponentially weighted moving average with a span of 10 for the specified columns within a player group.

    Args:
        player_group (pandas.DataFrame): DataFrame representing a group of player data.
        columns (list): List of column names to calculate the exponentially weighted moving average for.

    Returns:
        pandas.DataFrame: DataFrame containing the exponentially weighted moving average values for the specified columns, shifted by 1.

    """

    return player_group[columns].ewm(span=10, adjust = False).mean().shift(1)

def create_rolling_feature(player_stats_enriched, calculate_function, numeric_columns, suffix):
    """
    Creates a rolling feature by applying the specified calculate_function to the numeric_columns within each player group.

    Args:
        player_stats_enriched (pandas.DataFrame): DataFrame containing enriched player statistics.
        calculate_function (function): Function to calculate the rolling feature.
        numeric_columns (list): List of column names to calculate the rolling feature for.
        suffix (str): Suffix to append to the column names of the rolling feature.

    Returns:
        pandas.DataFrame: DataFrame containing the player statistics enriched with the rolling feature.

    """
    
    player_stats_rolling = player_stats_enriched.groupby('Player').apply(calculate_function, columns = numeric_columns).droplevel(level = 0)
    player_stats_rolling.columns = [f"{x}_{suffix}" for x in list(player_stats_rolling)]
    player_stats_enriched = player_stats_enriched.merge(player_stats_rolling, how = 'left', left_index=True, right_index=True)
    
    return player_stats_enriched

def create_player_stats_features(player_stats_enriched):
    """
    Creates additional player statistics features based on the given player_stats_enriched DataFrame.

    Args:
        player_stats_enriched (pandas.DataFrame): DataFrame containing enriched player statistics.

    Returns:
        pandas.DataFrame: DataFrame containing the player statistics enriched with additional features.

    """
    
    player_stats_enriched = player_stats_enriched.sort_values(by = 'Match_ID')
    
    player_stats_enriched['Games_Played'] = player_stats_enriched.groupby('Player').cumcount()
        
    player_stats_enriched = create_rolling_feature(player_stats_enriched, calculate_rolling_avg5, player_columns, suffix = "avg5")
    player_stats_enriched = create_rolling_feature(player_stats_enriched, calculate_rolling_avg10, player_columns, suffix = "avg10")
    player_stats_enriched = create_rolling_feature(player_stats_enriched, calculate_ewm5, player_columns, suffix = "ewm5")
    player_stats_enriched = create_rolling_feature(player_stats_enriched, calculate_ewm10, player_columns, suffix = "ewm10")

    return player_stats_enriched

def update_player_stats_enriched():
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
    
    season_list = list(player_stats_enriched_features['Season'].unique())
    for season in season_list:
        print(season)
        season_player_stats = player_stats_enriched_features[player_stats_enriched_features['Season'] == season]
        upload_data(season_player_stats, "Player_Stats_Enriched", overwrite=True, update_if_identical=True)
        
if __name__ == "__main__":
    update_player_stats_enriched()