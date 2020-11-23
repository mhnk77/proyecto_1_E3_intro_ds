from src.utils import load_df, save_df


def load_ingestion(path):
    return load_df(path)
 

def date_transformation(col,df):
 
    for i in col:
        data_clean[i] = data_clean[i].replace("/19$", "/2019", regex = True).replace("/18$", "/2018", regex = True, inplace = False)
    pass