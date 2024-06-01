
from chain_utils import get_match, get_teams, get_team
from mplfooty.pitch import VerticalPitch, Pitch
import matplotlib.pyplot as plt
from visualisation.plotting_pitches import plot_vertical_pitch_ax
from visualisation.afl_colours import team_colours
from matplotlib.colors import LinearSegmentedColormap


def get_home_away_chains(chains, match_id):
    
    match_chains = get_match(chains, match_id)
    home_team, away_team = get_teams(match_id)
    home_chains = get_team(match_chains, home_team)
    away_chains = get_team(match_chains, away_team)
    
    return home_chains, away_chains

def calculate_heatmap_statistics(pitch, home_chains, away_chains, bins = (4, 5)):
    
    home_offensive_value = pitch.bin_statistic(home_chains['left_right_start_x'], home_chains['left_right_start_y'], 
                                               values=home_chains['exp_offensive_value'], statistic='sum', bins=bins)
    away_offensive_value = pitch.bin_statistic(away_chains['left_right_start_x'], away_chains['left_right_start_y'], 
                                               values=away_chains['exp_offensive_value'], statistic='sum', bins=bins)

    relative_value_statistic = home_offensive_value['statistic'] - away_offensive_value['statistic']
    
    return relative_value_statistic
    
def create_colourmap(team_colours, home_team, away_team):
    
    home_colour = team_colours[home_team]['positive']
    away_colour = team_colours[away_team]['positive']
    
    colours = [away_colour, 'white', home_colour]
    custom_cmap = LinearSegmentedColormap.from_list('team_colour_scale', colors=colours)
    
    return custom_cmap

def plot_heatmap(pitch, ax, relative_value_statistic, match_chains, custom_cmap, bins = (4, 5), label = False, fontsize=10, fontcolour = "black"):
    
    plotting = pitch.bin_statistic(match_chains['left_right_start_x'], match_chains['left_right_start_y'], bins = bins)
    plotting['statistic'] = relative_value_statistic

    pitch.heatmap(plotting, ax=ax, cmap = custom_cmap, edgecolors="white", vmin = -50, vmax = 50)
    if label:
        pitch.label_heatmap(plotting, ax=ax, str_format='{:.0f}',
                                color=fontcolour, fontsize=fontsize, va='center', ha='center',
                                )
    
    return pitch, ax

def plot_expected_vaep_heatmap(ax, chain_data, match_id, bins = (4, 5), label = False, fontsize = 10, fontcolour = "black"):
    
    home_chains, away_chains = get_home_away_chains(chain_data, match_id)
    pitch, ax = plot_vertical_pitch_ax(ax, line_zorder=2, line_width=1, line_alpha=0.8)
    relative_value_statistic = calculate_heatmap_statistics(pitch, home_chains, away_chains, bins = bins)
    home_team, away_team = get_teams(match_id)
    custom_cmap = create_colourmap(team_colours, home_team, away_team)
    
    pitch, ax = plot_heatmap(pitch, ax, relative_value_statistic, home_chains, custom_cmap, bins = bins, label = label, fontsize=fontsize, fontcolour = fontcolour)
    
    return ax