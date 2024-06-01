import numpy as np
from chain_utils import get_match, get_team
import seaborn as sns
from clustering.clustering import GMM
from visualisation.plotting_pitches import plot_half_vertical_pitch_ax
from visualisation.plotting_utils import add_ax_text

def get_inside50s(chains):
    
    return chains[(chains['Inside50'] == True) & (chains['Disposal'] == "effective")]

def get_inside50_data(chains, match_id, team):
    
    match_chains = get_match(chains, match_id=match_id)
    team_chains = get_team(match_chains, team=team)
    inside50s = get_inside50s(team_chains)
        
    return inside50s

def create_inside50_plot_data(chains, match_id, team, clusters):
    
    inside50s = get_inside50_data(chains, match_id=match_id, team=team)
    inside50s['GMM_'+str(clusters)] = GMM(clusters=clusters, data=inside50s[['left_right_start_x', 'left_right_start_y', 'left_right_end_x', 'left_right_end_y']])
    
    return inside50s

def plot_arrows_clusters_pitch_ax(pitch, ax, data, start_xy, end_xy, colour_cycle, label = "label", top_n = 3):
                
    for rank in np.linspace(0, top_n-1, top_n):
        cluster = data[label].value_counts().index[rank]     
        clustered = data.loc[data[label] == cluster]
        pitch.arrows(clustered[start_xy[0]], clustered[start_xy[1]], clustered[end_xy[0]], clustered[end_xy[1]], color = colour_cycle[int(rank)], ax=ax, 
                     width=1, vertical = True)
    
    return pitch, ax

def plot_vertical_pitch_team_inside50s(ax, chain_data, match_id, team, clusters=10, top_n=3):
    
    inside50s = create_inside50_plot_data(chain_data, match_id, team, clusters)
    pitch, ax = plot_half_vertical_pitch_ax(ax, pad_bottom=-20)
    pitch, ax = plot_arrows_clusters_pitch_ax(pitch, ax, data=inside50s, 
                                            start_xy=('left_right_start_x', 'left_right_start_y'),
                                            end_xy = ('left_right_end_x', 'left_right_end_y'),
                                            colour_cycle=sns.color_palette("pastel", clusters),
                                            label='GMM_'+str(clusters), top_n=top_n)
    ax = add_ax_text(ax, 'Most Common Inside50s', team=team)
    return ax