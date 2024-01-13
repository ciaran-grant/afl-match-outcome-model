import pandas as pd
import numpy as np

def convert_home_away_to_team_opp_data(data):
    home_data = data.copy()
    home_data = home_data.rename(columns={"Home_Team": "Team", "Away_Team": "Opponent"})
    home_data["Home"] = 1
    home_data["Result"] = np.where(home_data["Home Win"] == 1, 1, 0)

    away_data = data.copy()
    away_data = away_data.rename(columns={"Home_Team": "Opponent", "Away_Team": "Team"})
    away_data["Home"] = 0
    away_data["Result"] = np.where(away_data["Home Win"] == 1, 0, 1)
    away_data["Margin"] = -1 * away_data["Margin"]

    team_opponent_data = pd.concat([home_data, away_data], axis=0)
    team_opponent_data = team_opponent_data.sort_values(
        by=["Match_ID", "Date"]
    ).reset_index(drop=True)

    team_opponent_data = team_opponent_data.drop(
        columns=["Home Win", "Attendance", "Weather_Type", "Round_ID", "Season"]
    )

    return team_opponent_data


def create_score_columns(data):
    data["Team_Score"] = np.where(
        data["Home"] == 1,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[-1]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[-1]).astype(int),
    )

    data["Opp_Score"] = np.where(
        data["Home"] == 0,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[-1]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[-1]).astype(int),
    )

    return data


def create_goal_columns(data):
    data["Team_Goals"] = np.where(
        data["Home"] == 1,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[0]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[0]).astype(int),
    )

    data["Opp_Goals"] = np.where(
        data["Home"] == 0,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[0]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[0]).astype(int),
    )
    return data


def create_behind_columns(data):
    data["Team_Behinds"] = np.where(
        data["Home"] == 1,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[1]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[1]).astype(int),
    )

    data["Opp_Behinds"] = np.where(
        data["Home"] == 0,
        data["Q4_Score"].apply(lambda x: x.split(" - ")[0].split(".")[1]).astype(int),
        data["Q4_Score"].apply(lambda x: x.split(" - ")[1].split(".")[1]).astype(int),
    )

    return data


def split_scores(data):
    data = create_score_columns(data)
    data = create_goal_columns(data)
    data = create_behind_columns(data)

    data = data.drop(columns=["Q4_Score"])

    return data

def rolling_averages(group, cols, new_cols, window = 3):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(window, closed='left').mean()
    group[new_cols] = rolling_stats
    return group[['Date', 'Team'] + new_cols]

def format_date_columns(data):
    
    data["Day"] = pd.to_datetime(data["Date"]).dt.day
    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["Year"] = pd.to_datetime(data["Date"]).dt.year
    
    date_cols = ['Day', 'Month', 'Year']

    return data, date_cols