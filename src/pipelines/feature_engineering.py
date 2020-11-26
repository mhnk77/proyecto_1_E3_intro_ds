from src.utils.utils import load_df, save_df
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit


def load_transformation(path):
    return load_df(path)


def feature_generation(df):

    return df


def feature_selection(df):

    x = df.drop(columns=['label'])
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=3)

    # ocuparemos un RF
    rfc = RandomForestClassifier(oob_score=True, random_state=42)
    # separando en train, test

    # definicion de los hiperparametros que queremos probar
    param_grid = {
        'n_estimators': [200],
        'max_features': [5, 6, 7, 8, 9],
        'max_depth': [20, 25],
        'criterion': ['gini']}

    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=param_grid,
                          cv=tscv,
                          scoring='precision',
                          n_jobs=-1)

    start_time = time.time()
    cv_rfc.fit(x_train, y_train)
    print("Tiempo en ejecutar: ", time.time() - start_time)

    d_imp = {'feature': x_train.columns, 'importance': cv_rfc.feature_importances_}

    df_imp = pd.DataFrame(d_imp).sort_values(by='importance', ascending=False)

    imp_fac = 0.07
    m = df_imp[df_imp.importance > imp_fac]

    lt = []
    for i in m.feature:

        if i.find('inc_') >= 0:
            lt.append('inc_')
        elif i.find('te_') >= 0:
            lt.append('te_')

        elif i.find('del_') >= 0:
            lt.append('del_')

        elif i.find('sin_hr') >= 0:
            lt.append('sin_hr')
            lt.append('cos_hr')

        elif i.find('cos_hr') >= 0:
            lt.append('sin_hr')
            lt.append('cos_hr')

        elif i.find('sin_dia') >= 0:
            lt.append('sin_dia')
            lt.append('cos_dia')

        elif i.find('cos_dia') >= 0:
            lt.append('sin_dia')
            lt.append('cos_dia')

        elif i.find('sin_mes') >= 0:
            lt.append('sin_mes')
            lt.append('cos_mes')

        elif i.find('cos_mes') >= 0:
            lt.append('sin_mes')
            lt.append('cos_mes')

        elif i.find('ano') >= 0:
            lt.append('ano')

    s = pd.Series(lt)
    st = s.str.cat(sep='|')
    df = df.filter(regex=st)

    return df


def save_fe(df, path):
    save_df(df, path)


def feature_engineering(path):
    df = load_transformation(path)
    #df = feature_generation(df)
    df = feature_selection(df)
    save_fe(df, "output/fe_df.pkl")
