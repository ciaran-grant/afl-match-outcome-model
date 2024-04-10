import numpy as np

def count_gt50(x):
    return np.sum((x > 50)[::-1])

def count_gt80(x):
    return np.sum((x > 80)[::-1])

def count_gt100(x):
    return np.sum((x > 100)[::-1])

def ema3(x):
    alpha = 2 / (3 + 1)
    weights = (1 - alpha) ** (np.arange(len(x)) + 1)
    weights /= weights.sum()
    ema = np.convolve(x, weights, mode='full')[:len(x)]
    return ema[-1]

def ema5(x):
    alpha = 2 / (5 + 1)
    weights = (1 - alpha) ** (np.arange(len(x)) + 1)
    weights /= weights.sum()
    ema = np.convolve(x, weights, mode='full')[:len(x)]
    return ema[-1]

def ema10(x):
    alpha = 2 / (10 + 1)
    weights = (1 - alpha) ** (np.arange(len(x)) + 1)
    weights /= weights.sum()
    ema = np.convolve(x, weights, mode='full')[:len(x)]
    return ema[-1]

def ema20(x):
    alpha = 2 / (20 + 1)
    weights = (1 - alpha) ** (np.arange(len(x)) + 1)
    weights /= weights.sum()
    ema = np.convolve(x, weights, mode='full')[:len(x)]
    return ema[-1]

score_kwargs = {
    "lag_feature": {
        "lag": [1],
        "mean": [[1, 5]],
        "std": [[1, 5]],
        ema3: [[1, 5]],
        ema5: [[1, 5]],
        ema20: [[1, 20]],
    }
}
team_performance_cols = ['Team_Win', 'Team_Score', 'Team_Margin', 'Team_xScore_sum', 'Team_xScore_sum_Margin', 'Team_exp_vaep_value_sum', 'Team_exp_vaep_value_sum_Margin']
opp_performance_cols = ['Opponent_Score', 'Opponent_xScore_sum', 'Opponent_exp_vaep_value_sum']

player_kwargs = {
    "lag_feature": {
     ema5: [[1, 5]],
    }
}
squad_performance_cols = ['xScore', 'exp_vaep_value']

diff_feature_list = [
    'Distance_Travelled',
    'ELO','ELO_probs',
    'xELO', 'xELO_probs',
    'Squad_xScore_ema5_1_5', 'Squad_exp_vaep_value_ema5_1_5',
    'For_Win_mean_1_5', 'For_Win_ema5_1_5', 'For_Win_ema20_1_20',
    'For_Score_mean_1_5', 'For_Score_ema5_1_5', 'For_Score_ema20_1_20',
    'For_Margin_mean_1_5', 'For_Margin_ema5_1_5', 'For_Margin_ema20_1_20',
    'For_xScore_sum_mean_1_5', 'For_xScore_sum_ema5_1_5', 'For_xScore_sum_ema20_1_20',
    'For_xScore_sum_Margin_mean_1_5', 'For_xScore_sum_Margin_ema5_1_5', 'For_xScore_sum_Margin_ema20_1_20',
    'For_exp_vaep_value_sum_mean_1_5', 'For_exp_vaep_value_sum_ema5_1_5', 'For_exp_vaep_value_sum_ema20_1_20',
    'For_exp_vaep_value_sum_Margin_mean_1_5', 'For_exp_vaep_value_sum_Margin_ema5_1_5', 'For_exp_vaep_value_sum_Margin_ema20_1_20'
]

modelling_features = [
    'Distance_Travelled_diff',
    'ELO_diff',
    'ELO_probs_diff',
    'xELO_diff',
    'xELO_probs_diff',
    'Squad_xScore_ema5_1_5_diff',
    'Squad_exp_vaep_value_ema5_1_5_diff',
    'For_Win_mean_1_5_diff',
    'For_Win_ema5_1_5_diff',
    'For_Win_ema20_1_20_diff',
    'For_Score_mean_1_5_diff',
    'For_Score_ema5_1_5_diff',
    'For_Score_ema20_1_20_diff',
    'For_Margin_mean_1_5_diff',
    'For_Margin_ema5_1_5_diff',
    'For_Margin_ema20_1_20_diff',
    'For_xScore_sum_mean_1_5_diff',
    'For_xScore_sum_ema5_1_5_diff',
    'For_xScore_sum_ema20_1_20_diff',
    'For_xScore_sum_Margin_mean_1_5_diff',
    'For_xScore_sum_Margin_ema5_1_5_diff',
    'For_xScore_sum_Margin_ema20_1_20_diff',
    'For_exp_vaep_value_sum_mean_1_5_diff',
    'For_exp_vaep_value_sum_ema5_1_5_diff',
    'For_exp_vaep_value_sum_ema20_1_20_diff',
    'For_exp_vaep_value_sum_Margin_mean_1_5_diff',
    'For_exp_vaep_value_sum_Margin_ema5_1_5_diff',
    'For_exp_vaep_value_sum_Margin_ema20_1_20_diff',
    'Distance_Travelled_ratio',
    'ELO_ratio',
    'ELO_probs_ratio',
    'xELO_ratio',
    'xELO_probs_ratio',
    'For_Win_mean_1_5_ratio',
    'For_Win_ema5_1_5_ratio',
    'For_Win_ema20_1_20_ratio',
    'For_Score_mean_1_5_ratio',
    'For_Score_ema5_1_5_ratio',
    'For_Score_ema20_1_20_ratio',
    'For_Margin_mean_1_5_ratio',
    'For_Margin_ema5_1_5_ratio',
    'For_Margin_ema20_1_20_ratio',
    'For_xScore_sum_mean_1_5_ratio',
    'For_xScore_sum_ema5_1_5_ratio',
    'For_xScore_sum_ema20_1_20_ratio',
    'For_xScore_sum_Margin_mean_1_5_ratio',
    'For_xScore_sum_Margin_ema5_1_5_ratio',
    'For_xScore_sum_Margin_ema20_1_20_ratio',
    'For_exp_vaep_value_sum_mean_1_5_ratio',
    'For_exp_vaep_value_sum_ema5_1_5_ratio',
    'For_exp_vaep_value_sum_ema20_1_20_ratio',
    'For_exp_vaep_value_sum_Margin_mean_1_5_ratio',
    'For_exp_vaep_value_sum_Margin_ema5_1_5_ratio',
    'For_exp_vaep_value_sum_Margin_ema20_1_20_ratio'
    ]