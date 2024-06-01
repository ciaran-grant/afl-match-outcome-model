from chain_utils import get_match, get_scores, get_teams
from visualisation.afl_colours import get_team_colours
import pandas as pd
import numpy as np
import seaborn as sns
from plottable import Table, ColumnDefinition


def create_match_score_summary(match_summary):
    match_summary = get_scores(match_summary)

    home_match_summary = match_summary[
        ["Home_Team", "Home_Goals", "Home_Behinds", "Home_Score"]
    ]
    home_match_summary.columns = ["Team", "Goals", "Behinds", "Score"]

    away_match_summary = match_summary[
        ["Away_Team", "Away_Goals", "Away_Behinds", "Away_Score"]
    ]
    away_match_summary.columns = ["Team", "Goals", "Behinds", "Score"]

    match_score_summary = pd.concat([home_match_summary, away_match_summary])

    return match_score_summary


def create_match_statistics_data(player_stats, score_summary):
    match_stats = (
        player_stats.groupby("Team")
        .sum()[
            [
                "Disposals",
                "Effective_Disposals",
                "Goal_Assists",
                "Pressure_Acts",
                "Shots_At_Goal",
                "Possessions",
                "xScore",
                "exp_offensive_value",
                "exp_vaep_value_received",
                "xDisposal",
            ]
        ]
        .reset_index()
    )
    match_stats = match_stats.merge(
        score_summary, how="left", left_on="Team", right_on="Team"
    )
    match_stats = match_stats.set_index("Team").T.reset_index()
    match_stats.rename(columns={"index": "Statistics"}, inplace=True)

    table_order = [
        "Score",
        "xScore",
        "Goals",
        "Behinds",
        "Shots_At_Goal",
        "Goal_Assists",
        "Possessions",
        "Disposals",
        "Effective_Disposals",
        "xDisposal",
        "exp_offensive_value",
        "exp_vaep_value_received",
        "Pressure_Acts",
    ]
    match_stats["Statistics"] = pd.Categorical(
        match_stats["Statistics"], categories=table_order, ordered=True
    )
    match_stats = match_stats.sort_values("Statistics").set_index("Statistics")

    return match_stats


def create_match_stats(summary, player_stats, match_id):
    match_summary = get_match(summary, match_id)
    match_player_stats = get_match(player_stats, match_id)
    match_score_summary = create_match_score_summary(match_summary)
    match_stats = create_match_statistics_data(match_player_stats, match_score_summary)
    home_team, away_team = get_teams(match_id)

    return match_stats[[home_team, away_team]].astype(float)


def get_normalised_match_stats(match_stats):
    return match_stats.div(match_stats.sum(axis=1), axis=0)


def plot_heatmap(ax, data):
    sns.heatmap(data, ax=ax, cbar=False, cmap="coolwarm")

    ax.set_xticklabels(list(data.columns))
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", which="both", top=False)

    ax.set_yticklabels(list(data.index))
    ax.tick_params(axis="y", which="both", left=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Match Statistics")

    return ax


def annotate_heatmap(ax, data):
    data_array = np.array(data)
    annotations = [
        ["{:.0f}", "{:.0f}"],
        ["{:.1f}", "{:.1f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.0f}", "{:.0f}"],
        ["{:.1f}", "{:.1f}"],
        ["{:.1f}", "{:.1f}"],
        ["{:.1f}", "{:.1f}"],
        ["{:.0f}", "{:.0f}"],
    ]

    for i in range(len(annotations)):
        for j in range(len(annotations[0])):
            formatted_text = annotations[i][j].format(data_array[i][j])
            ax.text(
                j + 0.5,
                i + 0.5,
                formatted_text,
                ha="center",
                va="center",
                fontsize=10,
                color="white",
            )

    return ax


### Plottable
def rename_statistics(match_stats):
    match_stats.index = [
        "Score",
        "Exp. Score",
        "Goals",
        "Behinds",
        "Shots",
        "Assists",
        "Possessions",
        "Disposals",
        "Eff. Disposals",
        "Exp. Disposals",
        "Exp. Value",
        "Exp. Value Received",
        "Pressures",
    ]

    return match_stats


def format_table_numbers(match_stats):
    match_stats = match_stats.astype(object)

    # Integers
    integer_cols = [
        "Score",
        "Goals",
        "Behinds",
        "Shots_At_Goal",
        "Goal_Assists",
        "Possessions",
        "Disposals",
        "Effective_Disposals",
        "Pressure_Acts",
    ]
    match_stats.loc[integer_cols] = match_stats.loc[integer_cols].astype(int)

    # Floats
    float_cols = [
        "xScore",
        "xDisposal",
        "exp_offensive_value",
        "exp_vaep_value_received",
    ]
    match_stats.loc[float_cols] = match_stats.loc[float_cols].astype(float).round(1)

    return match_stats


def define_column_format(home_team, away_team):
    home_primary_colour, home_secondary_colour = get_team_colours(home_team)
    away_primary_colour, away_secondary_colour = get_team_colours(away_team)

    col_defs = [
        ColumnDefinition(
            name="index",
            title="",
            textprops={"ha": "left", "weight": "bold"},
            border="right",
        ),
        ColumnDefinition(
            name=home_team,
            textprops={
                "ha": "center",
                "color": home_secondary_colour,
                "size": 14,
            },
        ),
        ColumnDefinition(
            name=away_team,
            textprops={
                "ha": "center",
                "color": away_secondary_colour,
                "size": 14,
            },
        ),
    ]

    return col_defs


def plottable_match_statistics(ax, plottable_match_stats, col_defs):
    tab = Table(
        plottable_match_stats,
        ax=ax,
        column_definitions=col_defs,
        row_dividers=True,
        textprops={"ha": "center", "fontname": "Karla"},
        footer_divider=True,
    )

    return tab, ax


def plot_match_statistics_table(ax, summary, player_stats, match_id):
    match_stats = create_match_stats(summary, player_stats, match_id)
    plottable_match_stats = rename_statistics(format_table_numbers(match_stats))

    home_team, away_team = get_teams(match_id)
    home_primary_colour, home_secondary_colour = get_team_colours(home_team)
    away_primary_colour, away_secondary_colour = get_team_colours(away_team)
    col_defs = define_column_format(home_team, away_team)

    table, ax = plottable_match_statistics(ax, plottable_match_stats, col_defs)
    table.columns[home_team].set_facecolor(home_primary_colour)
    table.col_label_row.cells[1].rectangle_patch.set_facecolor(home_primary_colour)
    table.columns[away_team].set_facecolor(away_primary_colour)
    table.col_label_row.cells[2].rectangle_patch.set_facecolor(away_primary_colour)

    return table, ax
