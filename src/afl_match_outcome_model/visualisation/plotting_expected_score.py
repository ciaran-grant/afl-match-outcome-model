import matplotlib.pyplot as plt
import matplotlib
from visualisation.afl_colours import team_colourmaps
from chain_utils import get_match, get_team
from visualisation.plotting_pitches import plot_half_vertical_pitch_ax
from visualisation.plotting_utils import add_ax_text

def create_set_shot_indicator(chain_data):
    
    chain_data['Set_Shot'] = chain_data['Event_Type1'].apply(lambda x: ("Mark" in str(x)) or ("Free" in str(x)))
    
    return chain_data

def get_shots(chain_data):
    
    shots = chain_data[chain_data['Shot_At_Goal'] == True]
    
    return shots

def get_set_shots(chain_data):
    
    shots = chain_data[chain_data['Shot_At_Goal'] == True]
    set_shots = shots[shots['Set_Shot'] == True]
    
    return set_shots

def get_open_shots(chain_data):
    
    shots = chain_data[chain_data['Shot_At_Goal'] == True]
    open_shots = shots[shots['Set_Shot'] == False]
    
    return open_shots

def get_shot_outcome(shots, final_state):
    
    return shots[shots['Final_State'] == final_state]

def get_shot_data_dict(chains, match_id, team):
    
    chains = create_set_shot_indicator(chains)
    match_chains = get_match(chains, match_id=match_id)
    team_chains = get_team(match_chains, team=team)
    set_shots = get_set_shots(team_chains)
    open_shots = get_open_shots(team_chains)
    
    shot_data_dict = {
        'set_goals': get_shot_outcome(set_shots, "goal"),
        'set_behinds': get_shot_outcome(set_shots, "behind"),
        'set_misses':get_shot_outcome(set_shots, "miss"),
        'open_goals': get_shot_outcome(open_shots, "goal"),
        'open_behinds': get_shot_outcome(open_shots, "behind"),
        'open_misses': get_shot_outcome(open_shots, "miss")}
    
    return shot_data_dict

def plot_expected_score_map(pitch, ax, shot_data_dict, cmap, size_ratio = 3, ec = None):

    if ec is None:
        ec = matplotlib.rcParams['text.color']

    norm = plt.Normalize(vmin=0, vmax=6)
    
    pitch.scatter(shot_data_dict['set_misses']['x'], shot_data_dict['set_misses']['y'], ax=ax, s=(shot_data_dict['set_misses']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['set_misses']['xScore'])), alpha=0.3, marker="s")
    pitch.scatter(shot_data_dict['set_behinds']['x'], shot_data_dict['set_behinds']['y'], ax=ax, s=(shot_data_dict['set_behinds']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['set_behinds']['xScore'])), marker="s")
    pitch.scatter(shot_data_dict['set_goals']['x'], shot_data_dict['set_goals']['y'], ax=ax, s=(shot_data_dict['set_goals']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['set_goals']['xScore'])), ec=ec, marker="s")

    pitch.scatter(shot_data_dict['open_misses']['x'], shot_data_dict['open_misses']['y'], ax=ax, s=(shot_data_dict['open_misses']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['open_misses']['xScore'])), alpha=0.3)
    pitch.scatter(shot_data_dict['open_behinds']['x'], shot_data_dict['open_behinds']['y'], ax=ax, s=(shot_data_dict['open_behinds']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['open_behinds']['xScore'])))
    pitch.scatter(shot_data_dict['open_goals']['x'], shot_data_dict['open_goals']['y'], ax=ax, s=(shot_data_dict['open_goals']['xScore']**2)*size_ratio, c=cmap(norm(shot_data_dict['open_goals']['xScore'])), ec=ec)
    
    return pitch, ax

def calculate_total_team_expected_score(shot_data_dict):
    
    return sum([shot_data_dict[shot_type]['xScore'].sum() for shot_type in shot_data_dict.keys()])

def plot_vertical_pitch_team_expected_score(ax, chain_data, match_id, team, fontsize = 36, size_ratio=3, ec="black"):
    
    shot_data_dict = get_shot_data_dict(chain_data, match_id, team)
    pitch, ax = plot_half_vertical_pitch_ax(ax)
    pitch, ax = plot_expected_score_map(pitch, ax, shot_data_dict, cmap = team_colourmaps[team], size_ratio = size_ratio, ec=ec)
    
    team_expected_score = calculate_total_team_expected_score(shot_data_dict)
    ax = add_ax_text(ax, f'{team_expected_score:.1f} xS', team=team, fontsize=fontsize)
    
    return ax