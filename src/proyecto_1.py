from src.pipelines.ingestion import ingest
from src.pipelines.transformation import transform
from src.pipelines.feature_engineering import feature_engineering


def main():

    # editar con el path del archivo csv
    path_csv = "../data/incidentes-viales-c5.csv"
    path_ing = "../output/ingest_df.pkl"
    #path_tra = "../output/transformation_df.pkl"

    ingest(path_csv)
    transform(path_ing)
    #feature_engineering(path_tra)


if __name__ == "__main__":
    main()
