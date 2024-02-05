from AFLPy.AFLData_Client import load_data

def load_matches(dataset_name = 'AFLTables_Match_Summary'):
    return load_data(Dataset_Name=dataset_name)

def load_player_stats(dataset_name = 'Fryzigg_Player_Stats'):
    return load_data(Dataset_Name=dataset_name)

def load_venues(dataset_name = 'Venues'):
    return load_data(Dataset_Name=dataset_name)

def load_team_info(dataset_name = 'Team_Info'):
    
    home_info = load_data(Dataset_Name=dataset_name).rename(columns = {'Team':'Home_Team', 'Home_Ground_1':'Home_Team_Venue'})
    away_info = load_data(Dataset_Name=dataset_name).rename(columns = {'Team':'Away_Team', 'Home_Ground_1':'Away_Team_Venue'})

    return home_info, away_info

def load_match_outcome_features(dataset_name = 'CG_Match_Outcome_Features', match_id = None):
    return load_data(Dataset_Name=dataset_name, ID=match_id)