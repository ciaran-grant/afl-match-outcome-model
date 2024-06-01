import numpy as np
from chain_utils import get_match, get_team
import seaborn as sns
from clustering.clustering import GMM
from visualisation.plotting_pitches import plot_vertical_pitch_ax
from visualisation.plotting_utils import add_ax_text

def get_defensive50s(chains):
    
    start_in_defensive50 = ((chains['left_right_start_x'] - (-chains['Venue_Length']/2))**2 + (chains['left_right_start_y'])**2)**0.5 < 50
    end_out_defensive50 = ((chains['left_right_end_x'] - (-chains['Venue_Length']/2))**2 + (chains['left_right_end_y'])**2)**0.5 > 50
    
    return chains[start_in_defensive50 & end_out_defensive50 & (chains['Disposal'] == "effective")]

def get_defensive50_data(chains, match_id, team):
    
    match_chains = get_match(chains, match_id=match_id)
    team_chains = get_team(match_chains, team=team)
    defensive50s = get_defensive50s(team_chains)
        
    return defensive50s

def create_defensive50_plot_data(chains, match_id, team, clusters):
    
    defensive50s = get_defensive50_data(chains, match_id=match_id, team=team)
    defensive50s['GMM_'+str(clusters)] = GMM(clusters=clusters, data=defensive50s[['left_right_start_x', 'left_right_start_y', 'left_right_end_x', 'left_right_end_y']])
    
    return defensive50s

def plot_arrows_clusters_pitch_ax(pitch, ax, data, start_xy, end_xy, colour_cycle, label = "label", top_n = 3):
                
    for rank in np.linspace(0, top_n-1, top_n):
        cluster = data[label].value_counts().index[rank]     
        clustered = data.loc[data[label] == cluster]
        pitch.arrows(clustered[start_xy[0]], clustered[start_xy[1]], clustered[end_xy[0]], clustered[end_xy[1]], color = colour_cycle[int(rank)], ax=ax, 
                     width=1, vertical = True)
    
    return pitch, ax

def plot_vertical_pitch_team_defensive50s(ax, chain_data, match_id, team, clusters=10, top_n=3):
    
    defensive50s = create_defensive50_plot_data(chain_data, match_id, team, clusters)
    pitch, ax = plot_vertical_pitch_ax(ax)
    pitch, ax = plot_arrows_clusters_pitch_ax(pitch, ax, data=defensive50s, 
                                              start_xy=('left_right_start_x', 'left_right_start_y'),
                                              end_xy = ('left_right_end_x', 'left_right_end_y'),
                                              colour_cycle=sns.color_palette("pastel", clusters),
                                              label='GMM_'+str(clusters), top_n=top_n)
    ax = add_ax_text(ax, 'Most Common Kicks from D50', team=team)

    return ax