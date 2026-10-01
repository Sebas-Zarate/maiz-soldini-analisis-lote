"""
Paso 1 - Carga, limpieza y control de calidad
Proyecto: Diagnóstico de lote de maíz - Soldini, Santa Fe

Qué hace:
  - Carga los tres datasets de origen (monitor de cosecha, muestreo de suelo, clima semanal)
  - Revisa nulos, duplicados y valores fuera de rango agronómico
  - Detecta outliers de rendimiento con el criterio IQR
  - Genera el mapa de rendimiento sobre el polígono real del lote

Entradas (carpeta data/):  cosecha_monitor.csv, muestreo_suelo.csv, clima_semanal.csv, lote_soldini.geojson
Salidas  (carpeta outputs/): mapa_rendimiento.png, rendimiento_vs_velocidad.png

Uso:  python scripts/01_limpieza.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
cosecha = pd.read_csv(DATA / "cosecha_monitor.csv")
suelo = pd.read_csv(DATA / "muestreo_suelo.csv")
clima = pd.read_csv(DATA / "clima_semanal.csv", parse_dates=["fecha_inicio", "fecha_fin"])
lote = gpd.read_file(DATA / "lote_soldini.geojson")

for nombre, df in [("cosecha", cosecha), ("suelo", suelo), ("clima", clima)]:
    print(f"\n--- {nombre} ---")
    print("Filas x columnas:", df.shape)
    print(df.dtypes)
    print("Nulos por columna:")
    print(df.isna().sum())


# ---------------------------------------------------------------------------
# 2. Duplicados
# ---------------------------------------------------------------------------
print("\nDuplicados cosecha:", cosecha.duplicated().sum())
print("Duplicados suelo:", suelo.duplicated().sum())
print("Duplicados coords cosecha:", cosecha.duplicated(subset=["lat", "lon"]).sum())


# ---------------------------------------------------------------------------
# 3. Rangos agronómicos plausibles
# ---------------------------------------------------------------------------
checks = {
    "cosecha.rendimiento_kg_ha": cosecha["rendimiento_kg_ha"].between(1500, 16000),
    "cosecha.humedad_grano_pct": cosecha["humedad_grano_pct"].between(10, 30),
    "cosecha.velocidad_kmh": cosecha["velocidad_kmh"].between(1, 12),
    "suelo.pH": suelo["pH"].between(3.5, 9),
    "suelo.materia_organica_pct": suelo["materia_organica_pct"].between(0, 10),
    "clima.precipitacion_mm": clima["precipitacion_mm"].between(0, 200),
}
print("\nValores fuera de rango agronómico:")
hubo_fuera = False
for nombre, mask in checks.items():
    n_fuera = int((~mask).sum())
    if n_fuera:
        hubo_fuera = True
        print(f"  {nombre}: {n_fuera} valores fuera de rango")
if not hubo_fuera:
    print("  ninguno")


# ---------------------------------------------------------------------------
# 4. Outliers de rendimiento (criterio IQR)
# ---------------------------------------------------------------------------
q1, q3 = cosecha["rendimiento_kg_ha"].quantile([0.25, 0.75])
iqr = q3 - q1
lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = ~cosecha["rendimiento_kg_ha"].between(lim_inf, lim_sup)
print(f"\nOutliers IQR en rendimiento: {outliers.sum()} de {len(cosecha)} ({outliers.mean() * 100:.2f}%)")
print(f"Límites: {lim_inf:.0f} - {lim_sup:.0f} kg/ha")
# El outlier detectado se documenta pero se conserva: el análisis usa las 3.700 filas.


# ---------------------------------------------------------------------------
# 5. Rendimiento vs velocidad de cosecha (chequeo de sesgo del monitor)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(cosecha["velocidad_kmh"], cosecha["rendimiento_kg_ha"], s=3, alpha=0.3)
plt.xlabel("Velocidad (km/h)")
plt.ylabel("Rendimiento (kg/ha)")
plt.title("Rendimiento vs velocidad de cosecha")
plt.tight_layout()
plt.savefig(OUT / "rendimiento_vs_velocidad.png", dpi=200, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 6. Mapa de rendimiento sobre el polígono real del lote
# ---------------------------------------------------------------------------
gdf_cosecha = gpd.GeoDataFrame(
    cosecha,
    geometry=gpd.points_from_xy(cosecha["lon"], cosecha["lat"]),
    crs="EPSG:4326",
)

fig, ax = plt.subplots(figsize=(8, 7))
lote.boundary.plot(ax=ax, color="black", linewidth=1.5)
gdf_cosecha.plot(
    ax=ax, column="rendimiento_kg_ha", cmap="RdYlGn",
    markersize=4, legend=True, legend_kwds={"label": "Rendimiento (kg/ha)"},
)
ax.set_title("Rendimiento de maíz sobre el lote - Soldini")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
plt.tight_layout()
plt.savefig(OUT / "mapa_rendimiento.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nListo. Gráficos guardados en outputs/")
