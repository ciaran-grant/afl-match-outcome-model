import numpy as np

def get_numeric_columns_list(data):
    
    return list(data.select_dtypes(np.number))

