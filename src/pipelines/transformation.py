from src.utils.utils import load_df, save_df
import pandas as pd
import numpy as np


def load_ingestion(path):
    return load_df(path)


def date_transformation(df, col):
    if col == "fecha_creacion":
        df[col] = df[col].replace("/19$", "/2019", regex=True) \
            .replace("/18$", "/2018", regex=True)
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y')

    elif col == "hora_creacion":
        z = df[df[col].str.match('^([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$') == False]
        df.loc[z.index, col] = pd.to_timedelta(z[col].astype(float), 'days')
        df[col] = pd.to_timedelta(df[col], 'ns')

    return df


def numeric_transformation(df, col):
    pass


def categoric_transformation(df, col):
    if col == "incidente_c4":
        df[col] = df[col].str.lower() \
            .str.normalize('NFKD').str.encode('ascii', errors='ignore') \
            .str.decode('utf-8').str.replace(' /', '', regex=True) \
            .str.replace('\-|\s', '_', regex=True)

        inc_top = df[col].value_counts() \
            .head(5).reset_index(name="n")['index'].values
        df['inc'] = np.where((df[col].isin(inc_top)),
                             df[col], "otros")
        df = pd.get_dummies(df, columns=['inc'])
        df = df.drop(columns=[col])

    elif col == "tipo_entrada":
        df[col] = df[col].str.lower() \
            .str.normalize('NFKD').str.encode('ascii', errors='ignore') \
            .str.decode('utf-8').str.replace(" ", "_", regex=True)

        tip_t = df[col].value_counts() \
            .head(4).reset_index(name="n")['index'].values
        df['te'] = np.where((df[col].isin(tip_t)),
                            df[col], "otros")
        df = pd.get_dummies(df, columns=['te'])
        df = df.drop(columns=[col])

    elif col == "del":
        df[col] = df[col].str.lower() \
            .str.normalize('NFKD').str.encode('ascii', errors='ignore') \
            .str.decode('utf-8').str.replace(" ", "_", regex=True)
        df = pd.get_dummies(df, columns=[col])

    elif col == "geopoint":
        geo_top = df[col].value_counts() \
            .reset_index(name="n").query('n >= 300')['index'].values
        df['geo_p'] = np.where((df[col].isin(geo_top)), "frecuente", "aislado")
        df = pd.get_dummies(df, columns=['geo_p'])
        df = df.drop(columns=[col])

    elif col == "fecha_creacion":
        df['ano'] = df[col].dt.year
        #df = pd.get_dummies(df, columns=['ano'])

    return df


def cyclic_transformation(df, col):
    if col == "hora_creacion":
        hours = 24
        df['sin_hr'] = np.sin(2 * np.pi * df[col].dt.components.hours / hours)
        df['cos_hr'] = np.cos(2 * np.pi * df[col].dt.components.hours / hours)
        df = df.drop(columns=[col])
    elif col == "fecha_creacion":
        dia = 7
        df['sin_dia'] = np.sin(2 * np.pi * df[col].dt.dayofweek + 1 / dia)
        df['cos_dia'] = np.cos(2 * np.pi * df[col].dt.dayofweek + 1 / dia)
        mes = 12
        df['sin_mes'] = np.sin(2 * np.pi * df[col].dt.month / dia)
        df['cos_mes'] = np.cos(2 * np.pi * df[col].dt.month / dia)
        df = df.drop(columns=[col])
    return df


def save_transformation(df, path):
    save_df(df, path)


def transform(path):
    print("transform")
    df = load_ingestion(path)

    #date_transform
    lt_date = ["fecha_creacion", "hora_creacion"]
    for i in lt_date:
        df = date_transformation(df, i)

    #categoric_transform
    lt_cat = ["incidente_c4", "tipo_entrada"]
    for i in lt_cat:
        df = categoric_transformation(df, i)

    ##delegacion_inicio
    df = categoric_transformation(df, "del")

    ##gepoint categorica
    df = categoric_transformation(df, "geopoint")

    ##se extrae año
    df = categoric_transformation(df, "fecha_creacion")

    #cyclic_transform
    lt_cyc = ["hora_creacion", "fecha_creacion"]
    for i in lt_cyc:
        df = cyclic_transformation(df, i)

    save_transformation(df, "../output/transformation_df.pkl")
