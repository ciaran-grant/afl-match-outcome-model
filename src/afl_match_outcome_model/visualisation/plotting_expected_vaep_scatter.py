import math
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from visualisation.afl_colours import get_team_colours
from chain_utils import get_match, get_team


def round_up_to_nearest_5(value):
    return math.ceil(value / 5) * 5


def get_match_expected_player_stats(player_stats, match_id, x, y):
    match_player_stats = get_match(player_stats, match_id)

    return match_player_stats[["Player", "Number", "Team", x, y]]


def get_team_expected_player_stats(player_stats, match_id, team, x, y):
    match_player_stats = get_match(player_stats, match_id)

    team_player_stats = get_team(match_player_stats, team)

    return team_player_stats[["Player", "Number", "Team", x, y]]


def get_diamond_plot_extents(data, x, y):
    xy_max = round_up_to_nearest_5(max(data[x].max(), data[y].max()))

    extent_dict = {}
    extent_dict["x_extent_min"] = -1
    extent_dict["x_extent_max"] = xy_max
    extent_dict["y_extent_min"] = -1
    extent_dict["y_extent_max"] = xy_max

    return extent_dict


def create_diamond_plot(fig, ax, extent_dict, nticks):
    
    transform = Affine2D().rotate_deg(45)
    plot_extents = extent_dict['x_extent_min'], extent_dict['x_extent_max'], extent_dict['y_extent_min'], extent_dict['y_extent_max']

    helper = floating_axes.GridHelperCurveLinear(transform, plot_extents,
                                                grid_locator1=MaxNLocator(nbins=nticks),
                                                grid_locator2=MaxNLocator(nbins=nticks)
                                                )

    ax_rotate = floating_axes.FloatingSubplot(fig, ax.get_subplotspec(), grid_helper=helper)
    fig.add_subplot(ax_rotate)
        
    return ax, ax_rotate, transform


def apply_diamond_plot_formatting(ax):
    ax.axis["bottom"].set_axis_direction("bottom")
    ax.axis["bottom"].major_ticklabels.set_axis_direction("right")
    ax.set_xlabel("Expected Disposals")

    ax.axis["left"].label.set_axis_direction("bottom")
    ax.set_ylabel("Expected Value")
    ax.grid(visible=True, lw=0.2, ls=":", color="lightgrey")

    return ax


def create_transformed_ax(ax, transform, extent_dict):
    aux_ax = ax.get_aux_axes(transform)
    aux_ax.vlines(x=15, ymin=-1, ymax=extent_dict["y_extent_max"], color="black", lw=0.5, ls="--")
    aux_ax.hlines(y=15, xmin=-1, xmax=extent_dict["x_extent_max"], color="black", lw=0.5, ls="--")

    return aux_ax


def plot_scatter(aux_ax, plot_data, x, y, team, s=100):
    primary_colour, _ = get_team_colours(team)
    aux_ax.scatter(plot_data[x], plot_data[y], c=primary_colour, s=s, zorder=2)

    return aux_ax

def plot_shirt_numbers(aux_ax, plot_data, x, y, team, numbersize = 20):
    _, secondary_colour = get_team_colours(team)

    for player in list(plot_data["Player"].unique()):
        player_data = plot_data[plot_data["Player"] == player].iloc[0]
        aux_ax.annotate(
            player_data["Number"],
            xy=(player_data[x], player_data[y]),
            ha="center",
            va="center",
            c=secondary_colour,
            size=numbersize,
            zorder=3,
        )

    return aux_ax

def remove_ax_spines_ticks(ax):
    
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax

def plot_match_expected_diamond_plot(fig, ax, player_stats, match_id, team, x, y, nticks = 10, markersize=100):
    
    team_expected_player_stats = get_team_expected_player_stats(player_stats, match_id, team, x, y)
    match_expected_player_stats = get_match_expected_player_stats(player_stats, match_id, x, y)
    
    extent_dict = get_diamond_plot_extents(match_expected_player_stats, x, y)
    
    ax, ax_rotate, transform = create_diamond_plot(fig, ax, extent_dict, nticks)
    ax_rotate = apply_diamond_plot_formatting(ax_rotate)
    
    aux_ax = create_transformed_ax(ax_rotate, transform, extent_dict)
    
    aux_ax = plot_scatter(aux_ax, plot_data=team_expected_player_stats, x = x, y = y, team = team, s=markersize)
    
    aux_ax = plot_shirt_numbers(aux_ax, plot_data=team_expected_player_stats, x = x, y = y, team = team, numbersize=markersize/40)
    
    ax = remove_ax_spines_ticks(ax)
    
    return ax
