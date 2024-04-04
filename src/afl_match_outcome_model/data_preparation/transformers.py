import pandas as pd
import numpy as np
import geopy.distance
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.series.summarize import WindowSummarizer

class YearRoundTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        X_transformed = X.copy()
        
        X_transformed['Year'] = X_transformed['Match_ID'].apply(lambda x: int(x.split("_")[1]))
        X_transformed['Round'] = X_transformed['Match_ID'].apply(lambda x: self.match_id_to_round_num(x))
        X_transformed['YearRound'] = X_transformed.apply(lambda row: int(str(row['Year']) + str(row['Round'])), axis=1)
        
        return X_transformed
    
    @staticmethod
    def match_id_to_round_num(match_id):
    
        finals_map = {
            2021: {
                'F1':24,
                'F2':25,
                'F3':26,
                'F4':27
            },
            2022: {
                'F1':24,
                'F2':25,
                'F3':26,
                'F4':27
            },
            2023: {
                'F1':25,
                'F2':26,
                'F3':27,
                'F4':28
            },
            2024:{
                'F1':25,
                'F2':26,
                'F3':27,
                'F4':28
            }
        }
        
        year, round_id = int(match_id.split("_")[1]), match_id.split("_")[2]
        return finals_map[year][round_id] if "F" in round_id else round_id
  
class DaysRestTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        
        unique_teams = list(set(list(X['Home_Team'].unique()) + list(X['Away_Team'].unique())))
        self.latest_date = {}
        for team in unique_teams:
            team_data = X[(X['Home_Team'] == team) | (X['Away_Team'] == team)]
            self.latest_date[team] = team_data['Date'].max()
        
        return self
    
    def transform(self, X):
    
        Xt = X.copy()
            
        # From the last match date, how many days
        Xt['Home_Days_Rest'] = Xt.apply(lambda row: (row["Date"] - self.latest_date[row['Home_Team']]).days, axis=1)
        Xt['Away_Days_Rest'] = Xt.apply(lambda row: (row["Date"] - self.latest_date[row['Away_Team']]).days, axis=1)
        
        return Xt
      
class ScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, score_col):
        self.score_col = score_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def parse_score(score):
            home, away = score.split(' - ')
            Home_Goals, Home_Behinds, Home_Score = map(int, home.split('.'))
            Away_Goals, Away_Behinds, Away_Score = map(int, away.split('.'))
            return Home_Goals, Home_Behinds, Home_Score, Away_Goals, Away_Behinds, Away_Score

        Xt = X.copy()
        Xt[['Home_Goals', 'Home_Behinds', 'Home_Score', 'Away_Goals', 'Away_Behinds', 'Away_Score']] = Xt[self.score_col].apply(parse_score).tolist()
             
        return Xt
    
class MarginTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy() 
        Xt['Margin'] = Xt['Home_Score'] - Xt['Away_Score']
        Xt['Home_Margin'] = Xt['Margin']
        Xt['Away_Margin'] = Xt['Margin']*-1
        return Xt
    
class WinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy() 
        Xt['Home_Win'] = np.where(Xt['Home_Margin'] > 0, 1, 0)
        Xt['Away_Win'] = np.where(Xt['Away_Margin'] > 0, 1, 0)
        return Xt
        
class ELOTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k_factor=32, initial_rating = 1500, expected = False):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.expected = expected
                
    def fit(self, X, y=None):
        self.elo_dict = {team: self.initial_rating for team in X['Home_Team'].unique()}
        elos, elo_probs = self._calculate_elo_ratings(X)
        self.elos = elos
        self.elo_probs = elo_probs
        return self

    def transform(self, X):
        X_elos, X_elo_probs = {}, {}
        for index, row in X.iterrows():
            match_id = row['Match_ID']
            if match_id in list(self.elos.keys()):
                X_elos[match_id] = self.elos[match_id]
                X_elo_probs[match_id] = self.elo_probs[match_id]
            else:
                home_team, away_team = self.get_home_team_from_match_id(match_id), self.get_away_team_from_match_id(match_id)
                X_elos[match_id] = [self.elo_dict[home_team], self.elo_dict[away_team]]
                
                home_elo_probs = self._calculate_elo_probability(self.elo_dict[home_team], self.elo_dict[away_team])
                X_elo_probs[match_id] = [home_elo_probs, 1-home_elo_probs]
                
        new_elo_df, new_elo_probs_df = self._convert_elo_dict_to_dataframe(X_elos, X_elo_probs)
        return self._merge_elo_ratings(X, new_elo_df, new_elo_probs_df)
    
    def _calculate_elo_probability(self, home_team_elo, away_team_elo):
        return 1 / (1 + 10 ** ((away_team_elo - home_team_elo) / 400))

    def _calculate_elo(self, home_team_elo, away_team_elo, margin):
        prob_win_home = self._calculate_elo_probability(home_team_elo, away_team_elo)
        score_diff = 1 if margin > 0 else 0.5 if margin == 0 else 0
        new_home_team_elo = home_team_elo + self.k_factor * (score_diff - prob_win_home)
        new_away_team_elo = away_team_elo + self.k_factor * ((1 - score_diff) - (1 - prob_win_home))
        return new_home_team_elo, new_away_team_elo

    def _calculate_elo_for_row(self, row):
        game_id, home_team, away_team = row['Match_ID'], row['Home_Team'], row['Away_Team']
        margin = row['Home_xScore_sum_Margin'] if self.expected else row['Margin']
        home_team_elo, away_team_elo = self.elo_dict[home_team], self.elo_dict[away_team]
        self.elo_dict[home_team], self.elo_dict[away_team] = self._calculate_elo(home_team_elo, away_team_elo, margin)
        return game_id, [home_team_elo, away_team_elo], [self._calculate_elo_probability(home_team_elo, away_team_elo), 1 - self._calculate_elo_probability(home_team_elo, away_team_elo)]

    def _calculate_elo_ratings(self, data):
        elos, elo_probs = {}, {}

        for _, row in data.iterrows():
            game_id, elos_game, elo_probs_game = self._calculate_elo_for_row(row)
            elos[game_id] = elos_game
            elo_probs[game_id] = elo_probs_game

        return elos, elo_probs

    def _convert_elo_dict_to_dataframe(self, elos, elo_probs):
        elo_columns = ['Home_xELO', 'Away_xELO'] if self.expected else ['Home_ELO', 'Away_ELO']
        elo_probs_columns = ['Home_xELO_probs', 'Away_xELO_probs'] if self.expected else ['Home_ELO_probs', 'Away_ELO_probs']
        
        elo_df = pd.DataFrame.from_dict(elos, orient='index', columns=elo_columns).rename_axis('Match_ID').reset_index()
        elo_probs_df = pd.DataFrame.from_dict(elo_probs, orient='index', columns=elo_probs_columns).rename_axis('Match_ID').reset_index()
        
        return elo_df, elo_probs_df

    def _merge_elo_ratings(self, X, elo_df, elo_probs_df):
        X = pd.merge(X, elo_df, how='left', on='Match_ID')
        X = pd.merge(X, elo_probs_df, how='left', on='Match_ID')
        return X
    
    @staticmethod
    def get_home_team_from_match_id(match_id):

        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[3])

    @staticmethod
    def get_away_team_from_match_id(match_id):
        
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[4]) 

class VenueInfoMerger(BaseEstimator, TransformerMixin):
    def __init__(self, venue_info, home_info, away_info):
        self.venue_info = venue_info
        self.home_info = home_info
        self.away_info = away_info        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_venue = self.merge_venue_info(X)
        X_home_away_venue =  self.merge_home_away_venue(X_venue)
        
        X_home_away_venue['Home_Distance_Travelled'] = X_home_away_venue.apply(lambda x: geopy.distance.geodesic((x['Venue_Latitude'], x['Venue_Longitude']), (x['Home_Team_Venue_Latitude'], x['Home_Team_Venue_Longitude'])).km, axis=1)
        X_home_away_venue['Away_Distance_Travelled'] = X_home_away_venue.apply(lambda x: geopy.distance.geodesic((x['Venue_Latitude'], x['Venue_Longitude']), (x['Away_Team_Venue_Latitude'], x['Away_Team_Venue_Longitude'])).km, axis=1)
        
        return X_home_away_venue
    
    def merge_venue_info(self, X):
        
        X_venue = X.merge(self.venue_info[['Venue', 'Latitude', 'Longitude']], how = 'left', on = 'Venue')
        X_venue = X_venue.rename(columns={'Latitude':'Venue_Latitude', 'Longitude':"Venue_Longitude"})
        
        return X_venue

    def merge_home_away_venue(self, X):
        # sourcery skip: inline-immediately-returned-variable
    
        X_home = X.merge(self.home_info[['Home_Team', 'Home_Team_Venue']], how = 'left', on = 'Home_Team')
        X_home_away = X_home.merge(self.away_info[['Away_Team', 'Away_Team_Venue']], how = 'left', on = 'Away_Team')
        
        home_venue_info = self.venue_info.copy().rename(columns={'Venue':'Home_Team_Venue', 'Latitude':'Home_Team_Venue_Latitude', 'Longitude':"Home_Team_Venue_Longitude"})
        X_home_venue = X_home_away.merge(home_venue_info[['Home_Team_Venue', 'Home_Team_Venue_Latitude', 'Home_Team_Venue_Longitude']], how = 'left', on = 'Home_Team_Venue')

        away_venue_info = self.venue_info.copy().rename(columns={'Venue':'Away_Team_Venue', 'Latitude':'Away_Team_Venue_Latitude', 'Longitude':"Away_Team_Venue_Longitude"})
        X_home_away_venue = X_home_venue.merge(away_venue_info[['Away_Team_Venue', 'Away_Team_Venue_Latitude', 'Away_Team_Venue_Longitude']], how = 'left', on = 'Away_Team_Venue')
        
        return X_home_away_venue
    
class TeamStatsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_column='team', stat=None, column = None):
        if stat is None:
            stat = 'sum'
        self.groupby_column = groupby_column
        self.stat = stat
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_grouped = X.groupby(self.groupby_column)[[self.column]]
        aggregated_stats = X_grouped.agg(self.stat)
        aggregated_stats.columns = [f'{self.column}_{self.stat}']
        aggregated_stats = aggregated_stats.reset_index()
        
        aggregated_stats['Home_Team'] = aggregated_stats['Match_ID'].apply(lambda match_id: self.get_home_team_from_match_id(match_id))
        aggregated_stats['Away_Team'] = aggregated_stats['Match_ID'].apply(lambda match_id: self.get_away_team_from_match_id(match_id))
        aggregated_stats['HomeAway'] = np.where(aggregated_stats['Team'] == aggregated_stats['Home_Team'], f'Home_{self.column}_{self.stat}', f'Away_{self.column}_{self.stat}')

        aggregated_stats = aggregated_stats.pivot_table(index = "Match_ID", values=f'{self.column}_{self.stat}', columns='HomeAway').reset_index()
        aggregated_stats.columns.name = ""
        aggregated_stats = aggregated_stats[['Match_ID', f'Home_{self.column}_{self.stat}', f'Away_{self.column}_{self.stat}']]
        aggregated_stats[f'Home_{self.column}_{self.stat}_Margin'] = aggregated_stats[f'Home_{self.column}_{self.stat}'] - aggregated_stats[f'Away_{self.column}_{self.stat}']
        aggregated_stats[f'Away_{self.column}_{self.stat}_Margin'] = aggregated_stats[f'Away_{self.column}_{self.stat}'] - aggregated_stats[f'Home_{self.column}_{self.stat}']
        
        return aggregated_stats
    
    @staticmethod
    def get_home_team_from_match_id(match_id):
    
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[3])

    @staticmethod
    def get_away_team_from_match_id(match_id):
        
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[4])   
        
class ExpectedMerger(BaseEstimator, TransformerMixin):
    def __init__(self, expected_score, expected_vaep):
        self.expected_score = expected_score
        self.expected_vaep = expected_vaep

    def fit(self, X, y=None):
        
        xscore_aggregator = TeamStatsAggregator(groupby_column=['Match_ID','Team'], stat='sum', column='xScore')
        self.xscore_team_sum = xscore_aggregator.fit_transform(self.expected_score)
        
        xvaep_aggregator = TeamStatsAggregator(groupby_column=['Match_ID','Team'], stat='sum', column='exp_vaep_value')
        self.xvaep_team_sum = xvaep_aggregator.fit_transform(self.expected_vaep)
        
        return self

    def transform(self, X):
        X_score = X.merge(self.xscore_team_sum, how = "left", on = "Match_ID")
        return X_score.merge(self.xvaep_team_sum, how = "left", on = "Match_ID")

class PastPerformanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols, window_summarizer_kwargs = None, for_against = "For") -> None:
        super().__init__()
        self.target_cols = target_cols
        if window_summarizer_kwargs is None:
            window_summarizer_kwargs = {}
        self.window_summarizer_kwargs = window_summarizer_kwargs
        self.window_summarizer = WindowSummarizer(**self.window_summarizer_kwargs, target_cols=self.target_cols)
        
        self.for_against = for_against
        
    def fit(self, X, y=None):
        
        X_team_opp = self.convert_home_away_to_team_opp(X)
        self.window_summarizer.fit(X_team_opp[self.target_cols])
        
        return self
    
    def transform(self, X):
        
        X_team_opp = self.convert_home_away_to_team_opp(X)
        X_team_opp_transformed = self.window_summarizer.transform(X_team_opp[self.target_cols])
        X_team_opp_transformed['Match_ID'] = X_team_opp['Match_ID']
        X_home_away_transformed = self.convert_team_opp_to_home_away(X_team_opp_transformed)
    
        return X.merge(X_home_away_transformed, how = "left", on = ['Match_ID', 'YearRound', 'Home_Team', 'Away_Team'])
    
    @staticmethod
    def convert_home_away_to_team_opp_single_team(match_summary, team):
    
        home_data = match_summary[match_summary['Home_Team'] == team].copy()
        home_data.columns = [x.replace("Home_Team", "Team").replace("Home", "Team").replace("Away_Team", "Opponent").replace("Away", "Opponent") for x in home_data.columns]
        home_data['Home'] = 1

        away_data = match_summary[match_summary['Away_Team'] == team].copy()
        away_data.columns = [x.replace("Home_Team", "Opponent").replace("Home", "Opponent").replace("Away_Team", "Team").replace("Away", "Team") for x in away_data.columns]
        away_data['Home'] = 0
        if 'Margin' in list(away_data):
            away_data['Margin'] = -1*away_data['Margin']

        return (
            pd.concat([home_data, away_data], axis=0)
            .sort_values(by="Match_ID")
            .reset_index(drop=True)
        )
        
    def convert_home_away_to_team_opp(self, match_summary):
        team_opp_data_list = [self.convert_home_away_to_team_opp_single_team(match_summary, team) for team in list(set(list(match_summary['Home_Team'].unique()) + list(match_summary['Away_Team'].unique())))]
        return pd.concat(team_opp_data_list, axis=0).set_index(['Team', 'YearRound']).sort_index()

    @staticmethod
    def get_home_team_from_match_id(match_id):
    
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[3])

    @staticmethod
    def get_away_team_from_match_id(match_id):
        
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[4])

    def convert_team_opp_to_home_away(self, team_opp_data):
        
        team_opp_data = team_opp_data.reset_index(drop = False).sort_values(by = 'Match_ID')
        team_opp_data['Home_Team'] = team_opp_data['Match_ID'].apply(lambda match_id: self.get_home_team_from_match_id(match_id))
        team_opp_data['Away_Team'] = team_opp_data['Match_ID'].apply(lambda match_id: self.get_away_team_from_match_id(match_id))
        
        home_data = team_opp_data[team_opp_data['Team'] == team_opp_data['Home_Team']]
        home_data = home_data.drop(columns=['Team'])
        away_data = team_opp_data[team_opp_data['Team'] == team_opp_data['Away_Team']]
        away_data = away_data.drop(columns=['Team'])
        
        if self.for_against == "For":
            home_data.columns = [x.replace("Team_", "Home_For_") for x in list(home_data)]
            away_data.columns = [x.replace("Team_", "Away_For_") for x in list(away_data)]
        elif self.for_against == "Against":
            home_data.columns = [x.replace("Opponent_", "Home_Against_") for x in list(home_data)]
            away_data.columns = [x.replace("Opponent_", "Away_Against_") for x in list(away_data)]
        
        home_away_data = home_data.merge(away_data, how = "inner", on = ['Match_ID', 'YearRound', 'Home_Team', 'Away_Team'])
        home_away_data = home_away_data[['Match_ID', 'YearRound', 'Home_Team', "Away_Team"] + [x for x in list(home_away_data) if x not in ['Match_ID', 'YearRound', 'Home_Team', "Away_Team"]]]
            
        return home_away_data

class SquadPerformanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, squads, expected_score, expected_vaep, target_cols, window_summarizer_kwargs = None) -> None:
        super().__init__()
        self.squads = squads
        self.expected_score = expected_score
        self.expected_vaep = expected_vaep
        self.target_cols = target_cols
        if window_summarizer_kwargs is None:
            window_summarizer_kwargs = {}
        self.window_summarizer_kwargs = window_summarizer_kwargs
        self.window_summarizer = WindowSummarizer(**self.window_summarizer_kwargs, target_cols=self.target_cols)
                
    def fit(self, X, y=None):
        
        X_squads = self.aggregate_expected_squads()
        X_squads = self.reset_squads_index(X_squads)
        
        self.window_summarizer.fit(X_squads[self.target_cols])
        
        return self
    
    def transform(self, X):
        
        X_squads = self.aggregate_expected_squads()
        X_squads = self.reset_squads_index(X_squads)
        
        X_squads_transformed = self.window_summarizer.transform(X_squads[self.target_cols])
        X_squads_transformed.columns = [f'Team_Squad_{x}' for x in X_squads_transformed]
        X_squads_transformed[['Match_ID', 'Team']] = X_squads[['Match_ID', 'Team']]
        X_squads_transformed = X_squads_transformed.reset_index().drop(columns = ['index'])
    
        X_squads_home_away = self.convert_squad_team_to_home_away(X_squads_transformed)
    
        return X.merge(X_squads_home_away, how = "left", on = ['Match_ID', 'Home_Team', 'Away_Team'])

    def aggregate_expected_squads(self):
        
        player_expected_score = self.expected_score.groupby(['Match_ID', 'Team', 'Player']).sum()[['xScore']].reset_index()
        player_expected_vaep = self.expected_vaep.groupby(['Match_ID', 'Team', 'Player']).sum()[['exp_vaep_value']].reset_index()

        squads_xscore = self.squads.merge(player_expected_score, how = "left", on = ['Match_ID', 'Team', 'Player'])
        expected_squads = squads_xscore.merge(player_expected_vaep, how = "left", on = ['Match_ID', 'Team', 'Player'])
        
        expected_squads[['xScore', 'exp_vaep_value']] = expected_squads[['xScore', 'exp_vaep_value']].fillna(0)
        
        return expected_squads
    
    @staticmethod
    def reset_squads_index(expected_squads):
        return pd.concat([expected_squads[expected_squads['Player'] == player].reset_index(drop=True).reset_index(drop = False).set_index(['Player', 'index']).sort_index() for player in list(expected_squads['Player'].unique())], axis=0)
    
    @staticmethod
    def get_home_team_from_match_id(match_id):
    
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[3])

    @staticmethod
    def get_away_team_from_match_id(match_id):
        
        return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[4])

    def convert_squad_team_to_home_away(self, X):
        # sourcery skip: extract-duplicate-method
        
        team_squad_sums = X.groupby(['Match_ID', 'Team']).sum().reset_index()
        
        team_squad_sums['Home_Team'] = team_squad_sums['Match_ID'].apply(lambda match_id: self.get_home_team_from_match_id(match_id))
        team_squad_sums['Away_Team'] = team_squad_sums['Match_ID'].apply(lambda match_id: self.get_away_team_from_match_id(match_id))

        home_data = team_squad_sums[team_squad_sums['Team'] == team_squad_sums['Home_Team']]
        home_data = home_data.drop(columns=['Team'])
        home_data.columns = [x.replace("Team_", "Home_") for x in list(home_data)]

        away_data = team_squad_sums[team_squad_sums['Team'] == team_squad_sums['Away_Team']]
        away_data = away_data.drop(columns=['Team'])
        away_data.columns = [x.replace("Team_", "Away_") for x in list(away_data)]

        home_away_data = home_data.merge(away_data, how = "inner", on = ['Match_ID', 'Home_Team', 'Away_Team'])
        home_away_data = home_away_data[['Match_ID', 'Home_Team', "Away_Team"] + [x for x in list(home_away_data) if x not in ['Match_ID', 'Home_Team', "Away_Team"]]]
        
        return home_away_data 
    
class HomeAwayDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a DataFrame with columns for both home and away features
        for feature in self.feature_list:
            X[f'{feature}_diff'] = X[f'Home_{feature}'] - X[f'Away_{feature}']

        return X

class HomeAwayRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a DataFrame with columns for both home and away features
        for feature in self.feature_list:
            X[f'{feature}_ratio'] = np.where(X[f'Away_{feature}'] == 0, 0, X[f'Home_{feature}'] / X[f'Away_{feature}'])
        return X

class ColumnFilter(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns = None, excluded_columns = None):
        self.selected = True
        self.excluded = True
        
        if selected_columns is None:
            self.selected = False
            selected_columns = []
        self.selected_columns = selected_columns
        
        if excluded_columns is None:
            self.excluded = False
            excluded_columns = []
        self.excluded_columns = excluded_columns
               
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.selected:
            return X[self.selected_columns]
        if self.excluded:
            return X.drop(columns = self.excluded_columns) 