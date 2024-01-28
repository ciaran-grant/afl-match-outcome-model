import pandas as pd
import numpy as np

def get_team_stats(match_stats, team):
    
    team_stats = match_stats[(match_stats['Home_Team'] == team) | (match_stats['Away_Team'] == team)]
    team_stats['Team'] = team
    team_stats = team_stats.sort_values(by = "Match_ID")
    
    return team_stats

def get_feature_ewm(data, feature_name, span):
    
    return data[feature_name].ewm(span=span).mean().shift(1)

def create_team_rolling_feature(match_stats, team, feature_name, span):
    team_stats = get_team_stats(match_stats, team)

    is_home_game = team_stats['Home_Team'] == team
    col_mapping = {
        'For': ('Home', 'Away'),
        'Against': ('Away', 'Home')
    }

    for suffix, (home_suffix, away_suffix) in col_mapping.items():
        home_col = f'{home_suffix}_{feature_name}'
        away_col = f'{away_suffix}_{feature_name}'

        team_stats[f'Team_{feature_name}_{suffix}'] = np.where(is_home_game, team_stats[home_col], team_stats[away_col])

        team_stats[f'Team_{feature_name}_{suffix}_ewm{span}'] = get_feature_ewm(team_stats, f'Team_{feature_name}_{suffix}', span)

    return team_stats

def create_all_teams_rolling_feature(match_stats, feature_name, span):

    team_list = sorted(set(match_stats['Home_Team']))

    return pd.concat([create_team_rolling_feature(match_stats, team, feature_name, span) for team in team_list], axis=0).sort_index()

 
def convert_team_rolling_feature_to_home_away(rolling_data, feature_name, span):

    rolling_match_stats = pd.DataFrame()
    for loc in ['Home', 'Away']:
        rename_dict = {
            f'Team_{feature_name}_For_ewm{span}': f'{loc}_{feature_name}_For_ewm{span}',
            f'Team_{feature_name}_Against_ewm{span}': f'{loc}_{feature_name}_Against_ewm{span}'
        }
        loc_data = rolling_data[rolling_data[f'{loc}_Team'] == rolling_data['Team']].rename(columns=rename_dict)
        rolling_match_stats = pd.concat([rolling_match_stats, loc_data], axis=1)

    return rolling_match_stats[[f'Home_{feature_name}_For_ewm{span}', f'Home_{feature_name}_Against_ewm{span}', f'Away_{feature_name}_For_ewm{span}', f'Away_{feature_name}_Against_ewm{span}']]

def create_all_teams_home_away_rolling_feature(match_stats, feature_name, span):
    
    rolling_data = create_all_teams_rolling_feature(match_stats, feature_name, span)
    return convert_team_rolling_feature_to_home_away(rolling_data, feature_name, span)
    
def create_team_rolling_features(match_stats, feature_list, span):
    
    for feature in feature_list:
        match_stats[
            [
                f'Home_{feature}_For_ewm{span}',
                f'Home_{feature}_Against_ewm{span}',
                f'Away_{feature}_For_ewm{span}',
                f'Away_{feature}_Against_ewm{span}',
            ]
        ] = create_all_teams_home_away_rolling_feature(
            match_stats, feature_name=feature, span=span
        )

    return match_stats