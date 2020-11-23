import pandas as pd
from src.utils.utils import save_df

def ingest_file(path_File):
    #'../data/incidentes-viales-c5.csv'
    df = pd.read_csv(path_File)
    return df

def drop_cols(df,cols):
    return

def generate_label(df):
    return df
 
def save_ingestion(df,path):
    save_df(df,path)
    return
  
def ingest(path):
    df = ingest_file(path)
    df = drop_cols(df)
    df = generate_label(df)
    save_ingestion(df,"../output/ingest_df.pkl")
    
    return