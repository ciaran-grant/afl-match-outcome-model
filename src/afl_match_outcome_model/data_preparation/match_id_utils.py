import re

round_map = {
    '00':0, 
    '01':1, 
    '02':2, 
    '03':3, 
    '04':4,
    '05':5,
    '06':6,
    '07':7,
    '08':8,
    '09':9,
    '10':10,
    '11':11,
    '12':12,
    '13':13,
    '14':14,
    '15':15,
    '16':16,
    '17':17,
    '18':18,
    '19':19,
    '20':20,
    '21':21,
    '22':22,
    '23':23,
    '24':24,
    'F1':25, 
    'F2':26, 
    'F3':27, 
    'F4':28,
    'F5':29
}

def get_competition_from_match_id(match_id):
    
    return match_id.split("_")[0]

def get_season_from_match_id(match_id):
    
    return int(match_id.split("_")[1])

def get_round_from_match_id(match_id):
    
    return round_map.get(match_id.split("_")[2])

def get_home_team_from_match_id(match_id):
    
    return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[3])

def get_away_team_from_match_id(match_id):
    
    return re.sub(r"(?<=\w)([A-Z])", r" \1", match_id.split("_")[4])

