import pandas as pd
import numpy as np
from src.utils.utils import save_df


def ingest_file(path_file):
    # '../data/incidentes-viales-c5.csv'
    df = pd.read_csv(path_file)
    return df


def drop_cols(df):

    #renombra delegacion_inicio por del
    df.rename(columns={'delegacion_inicio': 'del'}, inplace=True)
    list_cols = ["folio",
                 "dia_semana",
                 "fecha_cierre",
                 "año_cierre",
                 "mes_cierre",
                 "hora_cierre",
                 "clas_con_f_alarma",
                 "delegacion_cierre",
                 "codigo_cierre",
                 "latitud",
                 "longitud",
                 "mes"]
    df = df.drop(columns=list_cols)
    df.dropna(inplace=True)

    return df


def generate_label(df):
    cd = df["codigo_cierre"].str.extract('.*\(([A-Z])\).*')
    label = np.where((cd == "F") | (cd == "N"), 1, 0)
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
    save_ingestion(df, "../output/ingest_df.pkl")
    return
