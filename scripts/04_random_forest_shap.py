"""
Paso 4 - Modelo predictivo de rendimiento (Random Forest) + interpretación con SHAP
Proyecto: Diagnóstico de lote de maíz - Soldini, Santa Fe

Qué hace:
  - Entrena un Random Forest que predice el rendimiento a partir de pH y materia
    orgánica interpoladas, más latitud y longitud
  - Evalúa en un set de prueba (20%) con R² y MAE
  - Calcula valores SHAP para explicar cuánto y cómo aporta cada variable
  - Genera el resumen SHAP y los gráficos de dependencia de pH y materia orgánica

Entradas (outputs/): cosecha_con_suelo_interpolado.csv  (generado por el paso 2)
Salidas  (outputs/): metricas_modelo.json, shap_importance.csv, shap_values_test.csv,
                     shap_summary.png, shap_dependence_ph.png,
                     shap_dependence_materia_organica.png

Nota sobre el modelo: lat y lon se incluyen como predictores, por lo que el modelo
puede capturar estructura espacial que no está explicada por pH ni materia orgánica.
Las métricas son de un único split aleatorio (random_state=42): puntos vecinos del
monitor quedan en train y test, lo que tiende a ser optimista frente a una validación
espacial.

Uso:  python scripts/04_random_forest_shap.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

cosecha = pd.read_csv(OUT / "cosecha_con_suelo_interpolado.csv")

# ---------------------------------------------------------------------------
# 1. Datos de entrenamiento y prueba
# ---------------------------------------------------------------------------
features = ["pH_krig", "materia_organica_krig", "lat", "lon"]
X = cosecha[features]
y = cosecha["rendimiento_kg_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.1f} kg/ha")

# ---------------------------------------------------------------------------
# 3. SHAP
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Importancia global: valor absoluto medio de SHAP por variable
importancia = np.abs(shap_values).mean(axis=0)
print("\nImportancia SHAP (valor absoluto medio, kg/ha):")
for f, imp in sorted(zip(features, importancia), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.1f}")

# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.savefig(OUT / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

shap.dependence_plot("pH_krig", shap_values, X_test, feature_names=features, show=False)
plt.savefig(OUT / "shap_dependence_ph.png", dpi=200, bbox_inches="tight")
plt.close()

shap.dependence_plot("materia_organica_krig", shap_values, X_test,
                     feature_names=features, show=False)
plt.savefig(OUT / "shap_dependence_materia_organica.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. Exportar para el dashboard
# ---------------------------------------------------------------------------
pd.DataFrame({"feature": features, "importancia": importancia}).to_csv(
    OUT / "shap_importance.csv", index=False)

df_shap = X_test.copy()
for i, f in enumerate(features):
    df_shap[f"shap_{f}"] = shap_values[:, i]
df_shap.to_csv(OUT / "shap_values_test.csv", index=False)

with open(OUT / "metricas_modelo.json", "w") as f:
    json.dump({"r2": float(r2), "mae": float(mae)}, f, indent=2)

print("\nListo. Archivos guardados en outputs/")
