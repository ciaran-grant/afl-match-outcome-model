from visualisation.afl_colours import get_team_colours
from highlight_text import ax_text

def add_ax_text(ax, text, team, ha = "center", fontsize = 36):
    
    primary_colour, secondary_colour = get_team_colours(team) 
    
    ax_text(x = 0, y = 110, ha = ha, fontsize = fontsize,
        s=f'<{text}>',
        highlight_textprops=[
            {"bbox": {"edgecolor": primary_colour, "facecolor": primary_colour, "linewidth": 1.5, "pad": 5}, 
             "color": secondary_colour}
            ],
        ax=ax)
    
    return ax