import numpy as np
from chain_utils import get_match, get_teams
from visualisation.afl_colours import team_colours
from highlight_text import HighlightText


def get_expected_score_worm_data(chains, match_id):
    home_team, away_team = get_teams(match_id)
    match_chains = get_match(chains, match_id)
    match_scores = match_chains[~match_chains["xScore"].isna()]
    match_scores["net_xScore"] = np.where(
        match_scores["Team"] == home_team,
        match_scores["xScore"],
        -1 * match_scores["xScore"],
    )
    match_scores["cumsum_net_xScore"] = match_scores["net_xScore"].cumsum()

    return match_scores[["Duration", "cumsum_net_xScore"]]


def get_match_quarter_ends(chains, match_id):
    match_chains = get_match(chains, match_id)
    quarter_end_times = match_chains.groupby("Quarter").last()["Duration"]

    return {
        "first_quarter_end": quarter_end_times.loc[1],
        "second_quarter_end": quarter_end_times.loc[2],
        "third_quarter_end": quarter_end_times.loc[3],
        "fourth_quarter_end": quarter_end_times.loc[4],
    }


def create_expected_score_worm_ax(ax, data, match_id, quarter_end_times):
    fontsize = 16
    fontname = "Karla"
    ha, va = "center", "center"

    home_team, away_team = get_teams(match_id)

    ax.fill_between(
        data["Duration"],
        y1=data["cumsum_net_xScore"],
        where=(data["cumsum_net_xScore"] > 0),
        color=team_colours[home_team]["positive"],
    )
    ax.fill_between(
        data["Duration"],
        y1=data["cumsum_net_xScore"],
        where=(data["cumsum_net_xScore"] < 0),
        color=team_colours[away_team]["positive"],
    )

    biggest_lead = abs(data["cumsum_net_xScore"]).max()
    ax.set_ylim(-biggest_lead - 5, biggest_lead + 5)

    HighlightText(
        x=0,
        y=-biggest_lead,
        s="Start",
        ha=ha,
        va=va,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax,
    )
    HighlightText(
        x=quarter_end_times["first_quarter_end"],
        y=-biggest_lead,
        s="Q1",
        ha=ha,
        va=va,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax,
    )
    HighlightText(
        x=quarter_end_times["second_quarter_end"],
        y=-biggest_lead,
        s="Q2",
        ha=ha,
        va=va,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax,
    )
    HighlightText(
        x=quarter_end_times["third_quarter_end"],
        y=-biggest_lead,
        s="Q3",
        ha=ha,
        va=va,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax,
    )
    HighlightText(
        x=quarter_end_times["fourth_quarter_end"],
        y=-biggest_lead,
        s="Q4",
        ha=ha,
        va=va,
        fontname=fontname,
        fontsize=fontsize,
        ax=ax,
    )

    ax.axis("off")

    return ax


def plot_expected_score_worm(ax, chain_data, match_id):
    match_scores = get_expected_score_worm_data(chain_data, match_id)
    quarter_end_times = get_match_quarter_ends(chain_data, match_id)
    ax = create_expected_score_worm_ax(ax, match_scores, match_id, quarter_end_times)

    return ax
