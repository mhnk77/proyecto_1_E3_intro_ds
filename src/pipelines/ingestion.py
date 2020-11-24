import pandas as pd
import numpy as np
from src.utils.utils import save_df


def ingest_file(path_file):
    # '../data/incidentes-viales-c5.csv'
    df = pd.read_csv(path_file)
    return df


def drop_cols(df):
    list_cols = ["folio",
                 "fecha_cierre",
                 "año_cierre",
                 "mes_cierre",
                 "hora_cierre",
                 "clas_con_f_alarma",
                 "delegacion_cierre",
                 "codigo_cierre"]
    df = df.drop(columns=list_cols)
    return df


def generate_label(df):
    cd = df["codigo_cierre"].str.extract('.*\(([A-Z])\).*')
    label = np.where((cd == "F") | (cd == "N"), 0, 1)
    df = df.assign(label=label)
    return df


def save_ingestion(df, path):
    save_df(df, path)
    return


def ingest(path):
    print("ingest")
    df = ingest_file(path)
    df = generate_label(df)
    df = drop_cols(df)
    print(df.columns)
    save_ingestion(df, "../output/ingest_df.pkl")
    return
