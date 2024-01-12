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
        feature_list=None,
    ):
        self.categorical_features = categorical_features
        self.rolling_cols = rolling_cols
        self.new_rolling_cols = []
        self.rolling = rolling
        self.dummy_col_list = []
        self.dummy_features = []
        self.feature_list = feature_list

    def fit(self, X, y=None):
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)

        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_col_list.append(dummy_cols)
            self.dummy_features += list(dummy_cols)

        for col in self.rolling:
            self.new_rolling_cols += [f"{c}_rolling_{col}" for c in self.rolling_cols]

        self.feature_list = self.feature_list + self.new_rolling_cols + self.dummy_features

    def transform(self, X, y=None):
        matches = convert_home_away_to_team_opp_data(X)
        matches = split_scores(matches)

        for col in self.categorical_features:
            dummy_cols = pd.get_dummies(matches[col], prefix=col)
            self.dummy_features += list(dummy_cols)
        matches_dummies = pd.concat([matches] + self.dummy_col_list, axis=1)

        matches_rolling = matches_dummies.copy()
        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            self.new_rolling_cols += new_rolling_cols
            
            matches_rolling = matches_rolling.groupby("Team").apply(
                lambda x: rolling_averages(
                    x, self.rolling_cols, new_rolling_cols, window
                )
            ).reset_index(drop=True)

        return matches_rolling[self.feature_list]

    def get_response(self, X):
        matches = convert_home_away_to_team_opp_data(X)

        matches = split_scores(matches)

        matches_rolling = matches.copy()
        for window in self.rolling:
            new_rolling_cols = [f"{c}_rolling_{window}" for c in self.rolling_cols]
            self.new_rolling_cols += new_rolling_cols
            
            matches_rolling = matches_rolling.groupby("Team").apply(
                lambda x: rolling_averages(
                    x, self.rolling_cols, new_rolling_cols, window
                )
            ).reset_index(drop=True)

        return matches_rolling[ModellingDataContract.RESPONSE]
