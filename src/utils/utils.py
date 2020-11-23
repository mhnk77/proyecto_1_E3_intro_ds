## Python libraries
import pandas as pd
import unicodedata
import pickle


def load_df(path):
    data_pkl = pickle.load(open(path,"rb"))
    return data_pkl
  
  
def save_df(df,path):
    #'/full/path/to/file'
    with open(path, 'wb') as f:
         pickle.dump(df, f)
         f.close()
    return


   

 

