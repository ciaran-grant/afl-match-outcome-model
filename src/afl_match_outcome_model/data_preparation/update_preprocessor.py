import pandas as pd
import joblib
from AFLPy.AFLData_Client import load_data
from afl_match_outcome_model.predict.predict_margin import load_margin_preprocessor


def update_fit_new_expected_data(ID = None):
    
    preproc = load_margin_preprocessor()
    preproc = update_preprocessor_expected_data(preproc, ID = ID)
    preproc = fit_preprocessor(preproc)
    
    save_margin_preprocessor(preproc)
    
    return [True]

def update_fit_new_squads(ID = None):
    
    preproc = load_margin_preprocessor()
    preproc = update_preprocessor_new_squads(preproc, ID = ID)
    preproc = fit_preprocessor(preproc)
    
    save_margin_preprocessor(preproc)
    
    return [True]

def update_preprocessor_expected_data(preproc, ID = None):
    
    new_expected_score = load_data(Dataset_Name="CG_Expected_Score", ID = ID).sort_values(by = "Match_ID", ascending = True).reset_index()
    preproc['expected'].expected_score = pd.concat([preproc['expected'].expected_score, new_expected_score], axis = 0)
    preproc['squad'].expected_score = pd.concat([preproc['squad'].expected_score, new_expected_score], axis = 0)

    new_expected_vaep = load_data(Dataset_Name="CG_Expected_VAEP", ID = ID).sort_values(by = "Match_ID", ascending = True).reset_index()
    preproc['expected'].expected_vaep = pd.concat([preproc['expected'].expected_vaep, new_expected_vaep], axis = 0)
    preproc['squad'].expected_vaep = pd.concat([preproc['squad'].expected_vaep, new_expected_vaep], axis = 0)
    
    return preproc

def update_preprocessor_new_squads(preproc, ID = None):
    
    new_squads = load_data(Dataset_Name='AFL_API_Team_Positions', ID = ID).sort_values(by = "Match_ID", ascending = True)
    preproc['squad'].squads = pd.concat([preproc['squad'].squads, new_squads], axis = 0)
    
    return preproc

def fit_preprocessor(preproc):
    
    match_summary = load_data(Dataset_Name="AFL_API_Matches", ID = ["AFL", "2021", "2022", '2023', '2024']).sort_values(by = "Match_ID", ascending = True)
    match_summary = match_summary[match_summary['Match_Status'] == "CONCLUDED"]
    
    preproc.fit(match_summary)
    
    return preproc
    
def save_margin_preprocessor(preproc):
    
    preproc_file_path = "model_outputs/match_margin_pipeline_v10.joblib"
    
    return joblib.dump(preproc, preproc_file_path)