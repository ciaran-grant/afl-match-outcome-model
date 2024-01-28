import geopy.distance

def parse_score(score):
    """Parses a score string into goals, behinds, and total score.

    Args:
        score (str): The score string in the format 'goals - behinds'.

    Returns:
        tuple: A tuple containing the total score, goals, and behinds.
            - total (int): The total score.
            - goals (int): The number of goals.
            - behinds (int): The number of behinds.
    """
    goals, behinds, total = map(int, score.replace(" - ", ".").split("."))
    return total, goals, behinds

def create_score_features(match_stats):
    """Creates score-related features based on the match statistics.

    Args:
        match_stats (DataFrame): Input dataframe with match statistics.

    Returns:
        DataFrame: Updated dataframe with additional score-related features.
    """
    
    for team in ['Home', 'Away']:
        match_stats[f'{team}_Score'], match_stats[f'{team}_Goals'], match_stats[f'{team}_Behinds'] = zip(*match_stats['Q4_Score'].apply(lambda x: parse_score(x.split(" - ")[0 if team == 'Home' else 1])))
        match_stats[f'{team}_Scoring_Shots'] = match_stats[f'{team}_Goals'] + match_stats[f'{team}_Behinds']
        match_stats[f'{team}_Goal_Conversion'] = match_stats[f'{team}_Goals'] / match_stats[f'{team}_Scoring_Shots']

    return match_stats

def create_margin_features(match_stats):
    """Creates margin features based on the home and away scores.

    Args:
        match_stats (DataFrame): Input dataframe with match statistics.

    Returns:
        DataFrame: Updated dataframe with additional margin features.
    """

    match_stats['Home_Margin'] = match_stats['Home_Score'] - match_stats['Away_Score']
    match_stats['Away_Margin'] = -match_stats['Home_Margin']
    
    return match_stats

def create_win_features(match_stats):
    """Creates win features based on the margin of victory.

    Args:
        match_stats (DataFrame): Input dataframe with match statistics.

    Returns:
        DataFrame: Updated dataframe with additional win features.
    """
    
    match_stats['Home_Win'] = (match_stats['Home_Margin'] > 0).astype(int)
    match_stats['Away_Win'] = (match_stats['Away_Margin'] > 0).astype(int)

    return match_stats

def create_distance_travelled_feature(match_stats):
    match_stats['Home_Distance_Travelled'] = match_stats.apply(lambda x: geopy.distance.geodesic((x['Venue_Latitude'], x['Venue_Longitude']), (x['Home_Team_Venue_Latitude'], x['Home_Team_Venue_Longitude'])).km, axis=1)
    match_stats['Away_Distance_Travelled'] = match_stats.apply(lambda x: geopy.distance.geodesic((x['Venue_Latitude'], x['Venue_Longitude']), (x['Away_Team_Venue_Latitude'], x['Away_Team_Venue_Longitude'])).km, axis=1)

    return match_stats

def create_home_away_diff_feature(match_stats, feature_list):
    for feature in feature_list:
        match_stats[f"{feature}_diff"] = (match_stats[f'Home_{feature}']- match_stats[f'Away_{feature}'])
        
    return match_stats