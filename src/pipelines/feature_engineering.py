from src.utils.utils import load_df, save_df
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_transformation(path):
    return load_df(path)


def feature_generation(df):

    return df


def feature_selection(df):
    np.random.seed(42)
    x = df.drop(columns=['label'])
    y = df['label']

    # ocuparemos un RF
    classifier = RandomForestClassifier(oob_score=True, random_state=42)
    # separando en train, test
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # definicion de los hiperparametros que queremos probar
    hyper_param_grid = {'n_estimators': [500],
                        'max_depth': [10, 20],
                        'min_samples_split': [5, 10],
                        'max_features': [5, 6, 7, 8, 9]}

    # ocupemos grid search!
    gs = GridSearchCV(classifier,
                      hyper_param_grid,
                      scoring='precision',
                      cv=3,
                      n_jobs=-1)
    start_time = time.time()
    gs.fit(x_train, y_train)
    print("Tiempo en ejecutar: ", time.time() - start_time)

    return df


def save_fe(df, path):
    save_df(df, path)


def feature_engineering(path):
    df = load_transformation(path)
    df = feature_generation(df)
    df = feature_selection(df)
    save_fe(df, "../output/fe_df.pkl")
