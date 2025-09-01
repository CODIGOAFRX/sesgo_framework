GSMFair – Mini-framework local para mitigar sesgo en clasificación (CV screening)

GSMFair es una librería ligera y local (sin nube) para posprocesar las predicciones de un clasificador y reducir la disparidad entre grupos (p. ej., género) en FPR (false positive rate) y FNR (false negative rate).
Está pensada para demos y auditoría offline: no reentrena tu modelo; ajusta las salidas finales para equilibrar errores entre grupos y generar informes reproducibles.

✨ Características

Dos estrategias de mitigación:

EqualizeFprFnr → ajusta umbrales por grupo cuando existen scores/probabilidades.

equalize_rates_from_binary → reduce gaps sólo con predicción binaria (sin scores) “flipping” un mínimo de FP/FN por grupo hasta acercarlos al mejor grupo.

Integración plug-and-play con archivos existentes en reports/ (predicciones_thr.csv).

Informes listos para memoria/TFM: reports/metrics_genero.txt y CSV con predicciones mitigadas.

Funciona 100% en local (Windows/VS Code probado).

📦 Instalación

Requisitos: Python 3.9+ (recomendado virtualenv).

# Clona tu repo (o copia la carpeta)
# Crea y activa entorno (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instala dependencias y el paquete en modo editable
pip install -U pip
pip install -r requirements.txt   # si existe
pip install -e .


El paquete se instala como gsmfair desde src/.

📁 Estructura (mínima)
gms-cv/
├─ data/                         # (opcional) datasets originales
├─ models/                       # (opcional) modelos .joblib/.pkl
├─ reports/
│  └─ predicciones_thr.csv       # ENTRADA: género, label, preseleccionado (+ opcional score)
├─ src/
│  ├─ gsmfair/
│  │  └─ mitigation/
│  │     ├─ postprocessing.py    # EqualizeFprFnr + equalize_rates_from_binary
│  │     └─ __init__.py
│  └─ metrics_genero.py          # Script de auditoría+mitigación
└─ pyproject.toml


reports/predicciones_thr.csv debe contener como mínimo:

genero (H/M o 0/1), label (0/1), preseleccionado (0/1).

(Opcional) una columna de score: score, prob, prob_pos, y_score, etc.

🚀 Uso rápido

Asegúrate de tener reports/predicciones_thr.csv.

Ejecuta:

python src/metrics_genero.py


Revisa los resultados:

reports/metrics_genero.txt → Baseline vs Fair, por grupo y global.

reports/predicciones_thr_fair.csv → añade preseleccionado_fair con la salida mitigada.

Objetivo de demo: mostrar que FPR gap y FNR gap (diferencias entre H y M) disminuyen tras la mitigación.

🔧 Cómo funciona
A) Si tu CSV trae scores/probabilidades

Se usa EqualizeFprFnr:

from gsmfair.mitigation import EqualizeFprFnr
pp = EqualizeFprFnr(alpha=0.5, grid_size=401, clip=(0.01, 0.99))
pp.fit(y_true_val=y_true, y_scores_val=y_scores, s_val=s, ref_threshold=0.5)
y_pred_fair = pp.predict(y_scores=y_scores, s=s)


alpha: 1.0 prioriza FPR, 0.0 prioriza FNR, 0.5 equilibra.

grid_size: resolución para buscar el umbral por grupo (↑ = más fino).

B) Si tu CSV no trae scores (solo 0/1)

Se usa equalize_rates_from_binary:

from gsmfair.mitigation import equalize_rates_from_binary
y_pred_fair, info = equalize_rates_from_binary(
    y_true=y_true, y_pred=y_pred_base, s=s, alpha=0.5, seed=7
)


Qué hace: reduce FPR en el grupo con exceso de FP (cambiando algunos FP → 0) y reduce FNR en el grupo con exceso de FN (cambiando algunos FN → 1), hasta acercarlos al mejor grupo.

Ventaja: no necesita probabilidades; válido para auditoría offline.

⚙️ Parámetros recomendados

ALPHA en metrics_genero.py:

0.3 → prioriza bajar FNR gap (recuperar positivos).

0.8 → prioriza bajar FPR gap (evitar falsos positivos).

0.5 → equilibrio (valor por defecto).

GRID=401 (aumenta a 801 si usas EqualizeFprFnr y quieres más precisión).

TREF=0.5 umbral global de referencia (para el método de umbral por grupo).

📊 Métricas clave en el informe

TP: verdaderos positivos; FP: falsos positivos; TN, FN.

Precision = TP / (TP+FP)

Recall (TPR) = TP / (TP+FN)

FPR = FP / (FP+TN)

FNR = FN / (FN+TP)

Gaps: |FPR_H − FPR_M| y |FNR_H − FNR_M| (objetivo: bajar).

✅ Ejemplo real (tus resultados)

Tras aplicar equalize_rates_from_binary sobre tus predicciones:

FPR gap: 0.1609 → 0.0804 (−50%)

FNR gap: 0.1435 → 0.0717 (−50%)

Globalmente, bajan FP y FN, suben precision, F1 y accuracy.

Se imprime además un resumen de “flips” por grupo, p. ej.:
{'0': {'flip_FP_to_0': 1829, 'flip_FN_to_1': 0}, '1': {'flip_FP_to_0': 0, 'flip_FN_to_1': 127}}

🧪 Reproducibilidad y buenas prácticas

Usa la misma semilla (seed) para obtener resultados idénticos.

Versiona los informes metrics_genero.txt y los CSV con sufijos de fecha.

Para informes académicos, guarda también el diff Baseline → Fair.

🛠️ Solución de problemas

“No encuentra columnas”: asegúrate de que predicciones_thr.csv tenga genero, label, preseleccionado.
Si tus nombres son distintos, renómbralos o adapta el parser.

“Tengo scores pero no cambia nada”: revisa que la columna de score sea continua y esté bien mapeada a la clase positiva (p. ej., prob_pos).

“Quiero reducir aún más FNR/FPR gap”: sube/baja ALPHA y, si usas EqualizeFprFnr, aumenta GRID.

🤝 Contribuir

Issues y PRs bienvenidos.

Estilo: PEP8, mensajes de commit descriptivos, tests mínimos si tocas mitigación.
