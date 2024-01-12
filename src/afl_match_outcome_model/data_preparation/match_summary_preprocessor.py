import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

from afl_match_outcome_model.data_preparation.preprocessing import (
    convert_home_away_to_team_opp_data,
    split_scores,
    rolling_averages,
)
from afl_match_outcome_model.modelling_data_contract import ModellingDataContract


class MatchSummaryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_features: List[str],
        rolling_cols: List[str],
        rolling=[3],
        extra_features=None,
    ):
        self.categorical_features = categorical_features
        self.rolling_cols = rolling_cols
        self.new_rolling_cols = []
        self.rolling = rolling
        self.dummy_col_list = []
        self.dummy_features = []
        self.extra_features = extra_features

    def fit(self, X, y=None):
        
        self.feature_list = None
        
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)

        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            self.new_rolling_cols += new_rolling_cols
            
            matches = matches.groupby("Team").apply(
                lambda x: rolling_averages(
                    x, self.rolling_cols, new_rolling_cols, window
                )
            ).reset_index(drop=True)
            
        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_features += list(dummy_cols)

        self.feature_list = self.extra_features + self.new_rolling_cols + self.dummy_features
        
        return self

    def transform(self, X, y=None):
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)

        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            
            matches = matches.groupby("Team").apply(
                lambda x: rolling_averages(
                    x, self.rolling_cols, new_rolling_cols, window
                )
            ).reset_index(drop=True)
            
        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_col_list.append(dummy_cols)
        
        matches_dummies = pd.concat([matches] + self.dummy_col_list, axis=1)

        return matches_dummies[self.feature_list]

    def get_response(self, X):
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)

        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            self.new_rolling_cols += new_rolling_cols
            
            matches = matches.groupby("Team").apply(
                lambda x: rolling_averages(
                    x, self.rolling_cols, new_rolling_cols, window
                )
            ).reset_index(drop=True)
            

        return matches[ModellingDataContract.RESPONSE]
