import pandas as pd
import os

# Ruta correcta al CSV: ...\gms-cv\data\adult.csv
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "adult.csv"))

print("Usando:", DATA_PATH)

df = pd.read_csv(DATA_PATH)  # usa la cabecera del CSV
print(df.head())
print("Filas:", len(df))
print("Columnas:", df.columns.tolist())
