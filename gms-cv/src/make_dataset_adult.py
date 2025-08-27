import pandas as pd
import os

# Rutas
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_PATH  = os.path.join(ROOT, "data", "adult.csv")
OUT_PATH = os.path.join(ROOT, "data", "cv_train.csv")

# Carga (adult.csv ya tiene cabecera)
df = pd.read_csv(IN_PATH)

# Normaliza nombres posibles de la columna de género
if "sex" in df.columns:
    sexo_col = "sex"
elif "gender" in df.columns:
    sexo_col = "gender"
else:
    raise ValueError("No encuentro la columna de género (sex/gender).")

# Limpiar etiqueta income (quita puntos y espacios) y convertir a 0/1
df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
df["label"] = (df["income"] == ">50K").astype(int)

# Mapear género a H/M
df["genero"] = df[sexo_col].map({"Male": "H", "Female": "M"}).fillna("H")

# Construir un texto tipo CV desde las columnas estructuradas
def build_text(r):
    age = int(r["age"]) if "age" in df.columns else ""
    edu = r.get("education", "")
    occ = r.get("occupation", "")
    ms  = r.get("marital-status", "")
    rel = r.get("relationship", "")
    hrs = int(r.get("hours-per-week", 0)) if "hours-per-week" in df.columns else 0
    nat = r.get("native-country", "")
    wc  = r.get("workclass", "")
    race = r.get("race", "")
    return (f"Edad {age}. Educación {edu}. Profesión {occ}. "
            f"Estado civil {ms}. Relación {rel}. Horas/semana {hrs}. "
            f"País {nat}. Raza {race}. Workclass {wc}.")

df["texto_cv"] = df.apply(build_text, axis=1)

# Asignar id y seleccionar columnas finales
df = df.reset_index(drop=True)
df["id_cv"] = df.index + 1
out = df[["id_cv", "texto_cv", "genero", "label"]]

# Guardar
out.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"OK → {OUT_PATH} | Filas: {len(out)}")
