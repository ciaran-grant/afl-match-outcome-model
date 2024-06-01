from highlight_text import HighlightText
import matplotlib.image as image
from datetime import datetime
from visualisation.afl_colours import get_team_colours
from chain_utils import get_teams, get_match


team_short_names_map = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Greater Western Sydney": "GWS",
    "Hawthorn ": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Bulldogs",
}


def get_match_summary_date(match_summary):
    # Convert the string to a datetime object
    date_object = datetime.strptime(match_summary["Date"].iloc[0], "%Y-%m-%d %H:%M:%S")

    # Format the datetime object as a string with the desired format
    formatted_date = date_object.strftime("%Y-%m-%d")
    formatted_time = date_object.strftime("%H:%M")

    return formatted_date, formatted_time


def get_match_summary_score(match_summary):
    return match_summary["Q4_Score"].iloc[0]


def get_match_summary_location(match_summary):
    return match_summary["Venue"].iloc[0], match_summary["City"].iloc[0]


def get_match_season(match_id):
    return match_id[:4]


def get_match_round(match_id):
    return match_id.split("_")[0][4:]


def get_team_logo(logo_file_path, team_name):
    return image.imread(logo_file_path + "/" + team_name + "_logo.png")


def plot_home_logo(ax, logo_file_path, home_team, inset_ax=[-0.05, 0.15, 0.4, 0.4]):
    home_logo = get_team_logo(logo_file_path, home_team)
    ax_home = ax.inset_axes(inset_ax)
    ax_home.imshow(home_logo)
    ax_home.axis("off")


def plot_away_logo(ax, logo_file_path, away_team, inset_ax=[0.65, 0.15, 0.4, 0.4]):
    away_logo = get_team_logo(logo_file_path, away_team)
    ax_away = ax.inset_axes(inset_ax)
    ax_away.imshow(away_logo)
    ax_away.axis("off")


def plot_match_summary(
    ax,
    home_team,
    away_team,
    venue,
    city,
    season,
    round_str,
    match_date,
    match_time,
    score,
    logo_file_path,
):
    home_primary_colour, home_secondary_colour = get_team_colours(home_team)
    away_primary_colour, away_secondary_colour = get_team_colours(away_team)

    home_team_short = team_short_names_map.get(home_team)
    away_team_short = team_short_names_map.get(away_team)

    home_name_text_props = [
        {
            "bbox": {
                "edgecolor": home_primary_colour,
                "facecolor": home_primary_colour,
                "linewidth": 1.5,
                "pad": 2,
            },
            "color": "w",
            "fontsize": 24,
        }
    ]
    HighlightText(
        x=0.15,
        y=0.8,
        ha="center",
        va="center",
        s=f"<{home_team_short.center(15)}>",
        highlight_textprops=home_name_text_props,
        fontname="Karla",
        ax=ax,
    )

    away_name_text_props = [
        {
            "bbox": {
                "edgecolor": away_primary_colour,
                "facecolor": away_primary_colour,
                "linewidth": 1.5,
                "pad": 2,
            },
            "color": "w",
            "fontsize": 24,
        }
    ]
    HighlightText(
        x=0.85,
        y=0.8,
        ha="center",
        va="center",
        s=f"<{away_team_short.center(15)}>",
        highlight_textprops=away_name_text_props,
        fontname="Karla",
        ax=ax,
    )

    HighlightText(
        x=0.5,
        y=0.8,
        ha="center",
        va="center",
        s=f"{venue}, {city}",
        fontname="Karla",
        color="grey",
        fontsize=12,
        ax=ax,
    )
    HighlightText(
        x=0.5,
        y=0.7,
        ha="center",
        va="center",
        s=f"{season} {round_str}",
        fontname="Karla",
        color="grey",
        fontsize=12,
        ax=ax,
    )
    HighlightText(
        x=0.5,
        y=0.6,
        ha="center",
        va="center",
        s=f"{match_date} @ {match_time}",
        fontname="Karla",
        color="grey",
        fontsize=12,
        ax=ax,
    )

    HighlightText(
        x=0.5,
        y=0.4,
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        s=f"<{score}>",
        fontname="DM Sans",
        ax=ax,
    )

    plot_home_logo(ax, logo_file_path, home_team)
    plot_away_logo(ax, logo_file_path, away_team)

    ax.axis("off")

    return ax


def plot_match_information(ax, match_id, summary, logo_file_path):
    match_summary = get_match(summary, match_id)

    home_team, away_team = get_teams(match_id)
    match_summary = get_match(summary, match_id)
    match_date, match_time = get_match_summary_date(match_summary)
    score = get_match_summary_score(match_summary)
    venue, city = get_match_summary_location(match_summary)
    season = get_match_season(match_id)
    round_str = get_match_round(match_id)

    ax = plot_match_summary(
        ax,
        home_team,
        away_team,
        venue,
        city,
        season,
        round_str,
        match_date,
        match_time,
        score,
        logo_file_path,
    )

    return ax
