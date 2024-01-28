import numpy as np

def aggregate_stats(team_stats, team_type, numeric_stats):
    team_sum = team_stats.groupby(["Match_ID", f"{team_type}_Team"]).agg(np.sum).reset_index()
    team_sum.columns = ['Match_ID', f"{team_type}_Team"] + [f"{team_type}_{x}" for x in numeric_stats]
    return team_sum

def aggregate_player_stats_to_team_stats(player_stats, numeric_player_stats):
    home_team_stats = player_stats[player_stats['Home']==1][['Match_ID', 'Home_Team'] + numeric_player_stats]
    away_team_stats = player_stats[player_stats['Home']==0][['Match_ID', 'Away_Team'] + numeric_player_stats]

    home_team_sum = aggregate_stats(home_team_stats, "Home", numeric_player_stats)
    away_team_sum = aggregate_stats(away_team_stats, "Away", numeric_player_stats)

    return home_team_sum.merge(away_team_sum, how='left', on='Match_ID')

