from sklearn.pipeline import Pipeline
from AFLPy.AFLData_Client import load_data

from afl_match_outcome_model.data_preparation.transformers import (
    YearRoundTransformer,
    ScoreTransformer, MarginTransformer, WinTransformer, VenueInfoMerger, 
    ELOTransformer, ExpectedMerger, PastPerformanceTransformer, SquadPerformanceTransformer,
    HomeAwayDifferenceTransformer, HomeAwayRatioTransformer, ColumnFilter
    )
from afl_match_outcome_model.data_preparation.pipeline_utils import (
    diff_feature_list, team_performance_cols, score_kwargs, opp_performance_cols, player_kwargs, squad_performance_cols
    )

modelling_features = [
    'Distance_Travelled_diff',
    'ELO_diff',
    'xELO_diff',
    'Squad_xScore_ema5_1_5_diff',
    'Squad_exp_vaep_value_ema5_1_5_diff',
    'For_xScore_sum_Margin_ema5_1_5_diff',
    'For_xScore_sum_Margin_ema20_1_20_diff',
    'For_exp_vaep_value_sum_Margin_ema5_1_5_diff',
    'For_exp_vaep_value_sum_Margin_ema20_1_20_diff',
    ]

def fitted_pipeline():
    
    match_summary = load_data(Dataset_Name="AFL_API_Matches", ID = ["AFL", "2021", "2022", '2023', '2024']).sort_values(by = "Match_ID", ascending = True)
    match_summary = match_summary[match_summary['Match_Status'] == "CONCLUDED"]

    squads = load_data(Dataset_Name='AFL_API_Team_Positions', ID = ["AFL", "2021", "2022", '2023', '2024']).sort_values(by = "Match_ID", ascending = True)

    venue = load_data(Dataset_Name='Venues')
    home_info = load_data(Dataset_Name='Team_Info').rename(columns={'Team': 'Home_Team', 'Home_Ground_1': 'Home_Team_Venue'})
    away_info = load_data(Dataset_Name='Team_Info').rename(columns={'Team': 'Away_Team', 'Home_Ground_1': 'Away_Team_Venue'})

    expected_score = load_data(Dataset_Name="CG_Expected_Score", ID = ["AFL", "2021", "2022", '2023', '2024']).sort_values(by = "Match_ID", ascending = True)
    expected_vaep = load_data(Dataset_Name="CG_Expected_VAEP", ID = ["AFL", "2021", "2022", '2023', '2024']).sort_values(by = "Match_ID", ascending = True)
        
    features_pipeline = Pipeline([
        ('yearround', YearRoundTransformer()),
        ('venue', VenueInfoMerger(venue, home_info, away_info)),
        ("score", ScoreTransformer(score_col="Q4_Score")),
        ("margin", MarginTransformer()),
        ('win', WinTransformer()),
        ("elo", ELOTransformer(k_factor=32, initial_rating=1500)),
        ('expected', ExpectedMerger(expected_score, expected_vaep)),
        ("xelo", ELOTransformer(k_factor=32, initial_rating=1500, expected=True)),
        ('history_for', PastPerformanceTransformer(target_cols=team_performance_cols, window_summarizer_kwargs=score_kwargs, for_against="For")),
        ('history_against', PastPerformanceTransformer(target_cols=opp_performance_cols, window_summarizer_kwargs=score_kwargs, for_against="Against")),
        ('squad', SquadPerformanceTransformer(squads, expected_score, expected_vaep, target_cols=squad_performance_cols, window_summarizer_kwargs=player_kwargs)),
        ('home_away_diff', HomeAwayDifferenceTransformer(diff_feature_list)),
        ('home_away_ratio', HomeAwayRatioTransformer(diff_feature_list)),
        ('features', ColumnFilter(selected_columns=modelling_features)),
        ]
    )

    features_pipeline.fit(match_summary)
    
    return features_pipeline