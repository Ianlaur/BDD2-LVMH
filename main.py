import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV LVMH et retourne un DataFrame pandas.
    """
    df = pd.read_csv(csv_path)
    return df


def main() -> None:
    # Modifie ce chemin si tu changes le nom ou l'emplacement du fichier
    csv_path = "data/LVMH_Realistic_Merged_CA001-100.csv"

    df = load_data(csv_path)

    # Affiche les 5 premières lignes pour vérifier
    print("Aperçu des données :")
    print(df.head())

    print("\nInfos générales :")
    print(df.info())


if __name__ == "__main__":
    main()

