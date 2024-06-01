import numpy as np
from chain_utils import get_match, get_team
import seaborn as sns
from clustering.clustering import GMM
from visualisation.plotting_pitches import plot_vertical_pitch_ax
from visualisation.plotting_utils import add_ax_text

def get_possession_start_end(chains):
    
    chains_no_shots = chains[chains['Shot_At_Goal'] != True]
    chain_start = chains_no_shots.groupby('Chain_Number').first()[['left_right_start_x', 'left_right_start_y']]
    chain_end = chains_no_shots.groupby('Chain_Number').last()[['left_right_end_x', 'left_right_end_y']]
    possessions = chain_start.merge(chain_end, left_index=True, right_index=True).reset_index()
    
    return possessions

def get_possession_data(chains, match_id, team):
    
    match_chains = get_match(chains, match_id=match_id)
    team_chains = get_team(match_chains, team=team)
    possessions = get_possession_start_end(team_chains)
        
    return possessions

def create_possession_clusters(chains, match_id, team, clusters):
    
    possessions = get_possession_data(chains, match_id=match_id, team=team)
    possessions['GMM_'+str(clusters)] = GMM(clusters=clusters, data=possessions[['left_right_start_x', 'left_right_start_y', 'left_right_end_x', 'left_right_end_y']])
    
    return possessions

def merge_clusters_to_chains(chains, possessions, clusters):
    chains_possessions = chains.merge(possessions[['Chain_Number', "GMM_"+str(clusters)]], how = 'left', on = 'Chain_Number')
    chains_possessions = chains_possessions[chains_possessions['Shot_At_Goal'] != True]
    
    return chains_possessions

def create_possession_plot_data(chains, match_id, team, clusters):
    
    possessions_clusters = create_possession_clusters(chains, match_id=match_id, team=team, clusters=clusters)
    match_chains = get_match(chains, match_id)
    team_chains = get_team(match_chains, team)
    chain_possessions = merge_clusters_to_chains(team_chains, possessions_clusters, clusters)
        
    return chain_possessions

def plot_possession_clusters_pitch_ax(pitch, ax, data, colour_cycle, label = "label", top_n = 3):
                
    for rank in np.linspace(0, top_n-1, top_n):
        cluster = data[label].value_counts().index[rank]     
        clustered = data.loc[data[label] == cluster]
        for chain_num in list(clustered['Chain_Number'].unique()):
            possession = clustered[clustered['Chain_Number'] == chain_num]
            pitch.arrows(possession['left_right_start_x'], possession['left_right_start_y'], 
                         possession['left_right_end_x'], possession['left_right_end_y'], ax=ax, 
                         color = colour_cycle[int(rank)], alpha = 0.8,
                         width=1, vertical = True)
    
    return pitch, ax

def plot_vertical_pitch_team_possessions(ax, chain_data, match_id, team, clusters=10, top_n=3):
    
    possessions = create_possession_plot_data(chain_data, match_id, team, clusters)
    pitch, ax = plot_vertical_pitch_ax(ax)
    pitch, ax = plot_possession_clusters_pitch_ax(pitch, ax, data=possessions, 
                                              colour_cycle=sns.color_palette("pastel", clusters),
                                              label='GMM_'+str(clusters), top_n=top_n)
    ax = add_ax_text(ax, 'Most Common Possessions', team=team)

    return ax