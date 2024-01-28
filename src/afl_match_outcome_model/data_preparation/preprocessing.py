import numpy as np

def filter_data_AFL(data):
    
    return data[data['Match_ID'].apply(lambda x: x.split("_")[0] == "AFL")]

def merge_venue_info(matches, venue_info):
    
    matches = matches.merge(venue_info[['Venue', 'Latitude', 'Longitude']], how = 'left', on = 'Venue')
    matches = matches.rename(columns={'Latitude':'Venue_Latitude', 'Longitude':"Venue_Longitude"})
    
    return matches

def merge_home_away_venue(matches, home_info, away_info, venue_info):
    
    matches = matches.merge(home_info[['Home_Team', 'Home_Team_Venue']], how = 'left', on = 'Home_Team')
    matches = matches.merge(away_info[['Away_Team', 'Away_Team_Venue']], how = 'left', on = 'Away_Team')
    
    home_venue_info = venue_info.copy().rename(columns={'Venue':'Home_Team_Venue', 'Latitude':'Home_Team_Venue_Latitude', 'Longitude':"Home_Team_Venue_Longitude"})
    matches = matches.merge(home_venue_info[['Home_Team_Venue', 'Home_Team_Venue_Latitude', 'Home_Team_Venue_Longitude']], how = 'left', on = 'Home_Team_Venue')

    away_venue_info = venue_info.copy().rename(columns={'Venue':'Away_Team_Venue', 'Latitude':'Away_Team_Venue_Latitude', 'Longitude':"Away_Team_Venue_Longitude"})
    matches = matches.merge(away_venue_info[['Away_Team_Venue', 'Away_Team_Venue_Latitude', 'Away_Team_Venue_Longitude']], how = 'left', on = 'Away_Team_Venue')
    
    return matches

def create_home_flag(data):
        
    return np.where(data['Team'] == data['Home_Team'], 1, 0)

def merge_match_summary_team_stats(matches, team_stats):
    
    return matches.merge(team_stats, how = 'left', on = ['Match_ID', 'Home_Team', 'Away_Team'])

def filter_draws(data):
    return data[data['Margin'] != 0]

def outlier_eliminator(df):
    # Eliminate Essendon 2016 games
    essendon_filter_criteria = ~(((df['Home_Team'] == 'Essendon') & (df['Year'] == 2016)) | ((df['Away_Team'] == 'Essendon') & (df['Year'] == 2016)))
    df = df[essendon_filter_criteria].reset_index(drop=True)

    return df

def min_year_filter_data(data, year):
    return data[data['Year'] >= int(year)]

def sort_match_stats(match_stats):
    
    return match_stats.sort_values(by = ['Date', 'Match_ID', 'Home_Team', 'Away_Team'])

def get_match(match_stats, match_id):
    return match_stats[match_stats['Match_ID'] == match_id]