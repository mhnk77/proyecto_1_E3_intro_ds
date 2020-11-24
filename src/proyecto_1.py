from src.pipelines.ingestion import ingest


def main():
    # editar con el path del archivo csv
    path = "../data/incidentes-viales-c5.csv"
    ingest(path)


if __name__ == "__main__":
    main()
