import matplotlib.pyplot as plt
from mplfooty.pitch import Pitch, VerticalPitch

def plot_pitch():
    
    pitch = Pitch(pitch_length=160, pitch_width=135)

    fig, ax = pitch.draw()
    
    return fig, ax, pitch

def plot_pitch_ax(ax):
    
    pitch = Pitch(pitch_length=160, pitch_width=135)
    pitch.draw(ax=ax)
    
    return pitch, ax
    
def plot_half_pitch(pad_bottom = -20):
    
    pitch = Pitch(pitch_length=160, pitch_width=135,
                half=True, pad_bottom=pad_bottom)

    fig, ax = pitch.draw()
    
    return fig, ax, pitch

def plot_half_pitch_ax(ax, pad_bottom = -20):
    
    pitch = Pitch(pitch_length=160, pitch_width=135,
                half=True, pad_bottom=pad_bottom)

    pitch.draw(ax=ax) 
    
    return pitch, ax   

def plot_vertical_pitch():
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135)

    fig, ax = pitch.draw()
    
    return fig, ax, pitch

def plot_vertical_pitch_ax(ax, line_zorder=2, line_width=1, line_alpha=1):
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135, line_zorder=line_zorder, line_width=line_width, line_alpha=line_alpha)

    pitch.draw(ax=ax)    
    
    return pitch, ax

def plot_half_vertical_pitch(pad_bottom = -20):
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135,
                          half=True, pad_bottom=pad_bottom)

    fig, ax = pitch.draw()
    
    return fig, ax, pitch

def plot_half_vertical_pitch_ax(ax, pad_bottom = -20):
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135,
                          half=True, pad_bottom=pad_bottom)
    pitch.draw(ax=ax)
    
    return pitch, ax

def get_pitch(vertical=False, half=False, pad_bottom = -20):
    
    if vertical & ~half:
        fig, ax, pitch = plot_vertical_pitch() 
    elif vertical & half:
        fig, ax, pitch = plot_half_vertical_pitch(pad_bottom=pad_bottom) 
    elif ~vertical & half:
        fig, ax, pitch = plot_half_pitch(pad_bottom=pad_bottom) 
    else:
        fig, ax, pitch = plot_pitch()
    
    return fig, ax, pitch

def plot_pitch_grid(ncols, nrows):
    
    pitch = Pitch(pitch_length=160, pitch_width=135)
        
    fig, axs = pitch.grid(ncols=ncols, nrows=nrows, 
                        grid_height=0.85, title_height=0.06, endnote_height=0.04, title_space=0.04, endnote_space=0.01,
                        axis=False)
    
    return fig, axs, pitch

def plot_half_pitch_grid(ncols, nrows, pad_bottom = -20):
    
    pitch = Pitch(pitch_length=160, pitch_width=135,
                          half=True, pad_bottom=pad_bottom)
        
    fig, axs = pitch.grid(ncols=ncols, nrows=nrows, 
                        grid_height=0.85, title_height=0.06, endnote_height=0.04, title_space=0.04, endnote_space=0.01,
                        axis=False)
    
    return fig, axs, pitch

def plot_vertical_pitch_grid(ncols, nrows):
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135)
        
    fig, axs = pitch.grid(ncols=ncols, nrows=nrows, 
                        grid_height=0.85, title_height=0.06, endnote_height=0.04, title_space=0.04, endnote_space=0.01,
                        axis=False)
    
    return fig, axs, pitch

def plot_half_vertical_pitch_grid(ncols, nrows, pad_bottom = -20):
    
    pitch = VerticalPitch(pitch_length=160, pitch_width=135,
                          half=True, pad_bottom=pad_bottom)
        
    fig, axs = pitch.grid(ncols=ncols, nrows=nrows, 
                        grid_height=0.85, title_height=0.06, endnote_height=0.04, title_space=0.04, endnote_space=0.01,
                        axis=False)
    
    return fig, axs, pitch

def get_pitch_grid(ncols, nrows, vertical=False, half=False, pad_bottom=-20):
    
    if vertical & ~half:
        fig, axs, pitch = plot_vertical_pitch_grid(ncols=ncols, nrows=nrows) 
    elif vertical & half:
        fig, axs, pitch = plot_half_vertical_pitch_grid(ncols=ncols, nrows=nrows) 
    elif ~vertical & half:
        fig, axs, pitch = plot_half_pitch_grid(ncols=ncols, nrows=nrows) 
    else:
        fig, axs, pitch = plot_pitch_grid(ncols=ncols, nrows=nrows)
    
    return fig, axs, pitch
