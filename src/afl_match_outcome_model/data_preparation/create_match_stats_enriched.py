import pandas as pd
from afl_match_outcome_model.data_preparation._create_match_stats import load_match_data, combine_match_summary, get_match_stats, create_match_stats_features

def create_match_stats_enriched() -> pd.DataFrame:
    """
    Updates the match statistics enriched data by performing the following steps:

    1. Loads the match data dictionary using the `load_match_data` function.
    2. Combines the match summary using the `combine_match_summary` function.
    3. Retrieves the match statistics using the `get_match_stats` function.
    4. Creates match statistics features using the `create_match_stats_features` function.
    5. Sorts the match statistics enriched data by 'Match_ID'.
    6. Uploads the updated data to the 'Match_Stats_Enriched' table, overwriting existing data if present and updating if the new data is identical.

    Examples:
        update_match_stats_enriched()
    """
    match_data_dict = load_match_data()
    match_stats_enriched = combine_match_summary(match_data_dict)

    match_stats_enriched = get_match_stats(match_stats_enriched)
    match_stats_enriched = create_match_stats_features(match_stats_enriched)
    match_stats_enriched = match_stats_enriched.sort_values(by='Match_ID')

    return match_stats_enriched
    