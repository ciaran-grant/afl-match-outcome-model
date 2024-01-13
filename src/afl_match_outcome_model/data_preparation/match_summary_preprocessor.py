import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

from afl_match_outcome_model.data_preparation.preprocessing import (
    convert_home_away_to_team_opp_data,
    split_scores,
    rolling_averages,
    format_date_columns
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
        self.rolling = rolling
        self.extra_features = extra_features

    def fit(self, X, y=None):
        self.feature_list = None

        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)
        
        matches, date_cols = format_date_columns(matches)

        self.new_rolling_cols = []
        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            self.new_rolling_cols += new_rolling_cols

        self.dummy_features = []
        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_features += list(dummy_cols)
            
        
        self.feature_list = (
            self.extra_features + self.new_rolling_cols + self.dummy_features + date_cols
        )

        return self

    def transform(self, X, y=None):

        X_copy = X.copy()

        matches = convert_home_away_to_team_opp_data(X_copy)
        matches = split_scores(matches)

        matches = matches.sort_values(by = ['Date', 'Team', 'Opponent']).reset_index(drop = True)

        matches, _ = format_date_columns(matches)

        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]

            rolling_data = (
                matches.groupby("Team")
                .apply(
                    lambda x: rolling_averages(
                        x, self.rolling_cols, new_rolling_cols, window
                    )
                )
                .reset_index(drop=True)
            )
            
            matches = matches.merge(rolling_data, how = "left", on = ["Date", 'Team'])


        self.dummy_col_list = []
        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_col_list.append(dummy_cols)
        matches_dummies = pd.concat(self.dummy_col_list, axis=1)

        matches_transformed = pd.concat(
            [matches, matches_dummies], axis=1
        )

        for col in self.feature_list:
            if col not in list(matches_transformed):
                matches_transformed[col] = 0
                
        matches_transformed = matches_transformed.dropna(subset = self.new_rolling_cols).reset_index(drop = True)

        return matches_transformed[self.feature_list]

    def get_response(self, X):
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)
        
        matches = matches.sort_values(by = ['Date', 'Team', 'Opponent']).reset_index(drop = True)

        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]

            rolling_data = (
                matches.groupby("Team")
                .apply(
                    lambda x: rolling_averages(
                        x, self.rolling_cols, new_rolling_cols, window
                    )
                )
                .reset_index(drop=True)
            )
            
            matches = matches.merge(rolling_data, how = "left", on = ["Date", 'Team'])

        matches = matches.dropna(subset = self.new_rolling_cols).reset_index(drop = True)
        
        return matches[ModellingDataContract.RESPONSE]
