import numpy as np
from visualisation.plotting_pitches import get_pitch, get_pitch_grid

def plot_clusters_pitch(k, data, x, y, colour_cycle, label = "label", vertical = False, half = False, alpha = 0.5, s=20, pad_bottom = -20):
                
    fig, ax, pitch = get_pitch(vertical=vertical, half=half, pad_bottom=pad_bottom)
    
    for clust in np.linspace(0, k-1, k):        
        clustered = data.loc[data[label] == clust]
        pitch.scatter(clustered[x], clustered[y], color=colour_cycle[int(clust)], alpha = alpha, s = s, ax=ax)
        
    return fig, ax

def plot_arrows_clusters_pitch_ax(pitch, ax, data, start_xy, end_xy, colour_cycle, label = "label", top_n = 3):
                
    for rank in np.linspace(0, top_n-1, top_n):
        cluster = data[label].value_counts().index[rank]     
        clustered = data.loc[data[label] == cluster]
        pitch.arrows(clustered[start_xy[0]], clustered[start_xy[1]], clustered[end_xy[0]], clustered[end_xy[1]], color = colour_cycle[int(rank)], ax=ax, 
                     width=1, vertical = True)
    
    return ax

def plot_clusters_pitch_grid(k, data, x, y, colour_cycle, label = "label", vertical = False, half = False, nrows=3, ncols = 3, alpha=0.5, s=20, pad_bottom = -20):
         
    fig, axs, pitch = get_pitch_grid(ncols=ncols, nrows=nrows, vertical=vertical, half=half, pad_bottom=pad_bottom)
        
    for clust, ax in zip(np.linspace(0, k-1, k), axs['pitch'].flat[:k]):
        clustered = data.loc[data[label] == clust]
        pitch.scatter(clustered[x], clustered[y], color=colour_cycle[int(clust)], alpha = alpha, s = s, ax=ax)
        
    return fig, axs

def plot_arrows_clusters_pitch_grid(k, data, start_xy, end_xy, colour_cycle, label = "label", vertical = False, half = False, nrows=3, ncols = 3, pad_bottom = -20):
         
    fig, axs, pitch = get_pitch_grid(ncols=ncols, nrows=nrows, vertical=vertical, half=half, pad_bottom=pad_bottom)
    
    for clust, ax in zip(np.linspace(0, k-1, k), axs['pitch'].flat[:k]):
        clustered = data.loc[data[label] == clust]
        pitch.arrows(clustered[start_xy[0]], clustered[start_xy[1]], clustered[end_xy[0]], clustered[end_xy[1]], color = colour_cycle[int(clust)], ax=ax, width=1)

    return fig, axs