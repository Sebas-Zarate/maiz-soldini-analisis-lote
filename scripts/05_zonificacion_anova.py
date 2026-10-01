"""
Paso 5 - Zonificación de manejo (KMeans) y validación estadística (ANOVA)
Proyecto: Diagnóstico de lote de maíz - Soldini, Santa Fe

Qué hace:
  - Agrupa los puntos del lote según pH y materia orgánica interpoladas (KMeans, k=3)
  - Justifica k con el método del codo
  - Ordena las zonas de menor a mayor pH medio: baja, media y alta fertilidad
  - Mapea las zonas de manejo sobre el polígono del lote
  - Verifica con ANOVA de una vía que el rendimiento difiere entre zonas y calcula
    el tamaño del efecto (eta cuadrado)

Entradas (data/):    lote_soldini.geojson
Entradas (outputs/): cosecha_con_suelo_interpolado.csv  (generado por el paso 2)
Salidas  (outputs/): dataset_final_dashboard.csv, metricas_anova.json,
                     metodo_codo.png, zonificacion_manejo.png

Nota: las zonas se definen con pH y materia orgánica (no con el rendimiento), y las
etiquetas se asignan ordenando los clusters por pH medio. El ANOVA compara luego el
rendimiento entre zonas. Con miles de puntos vecinos el p-valor es muy chico casi por
construcción; por eso conviene mirar también el tamaño del efecto (eta cuadrado).

Uso:  python scripts/05_zonificacion_anova.py
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

lote = gpd.read_file(DATA / "lote_soldini.geojson")
cosecha = pd.read_csv(OUT / "cosecha_con_suelo_interpolado.csv")

# ---------------------------------------------------------------------------
# 1. Variables de agrupamiento (estandarizadas)
# ---------------------------------------------------------------------------
vars_cluster = cosecha[["pH_krig", "materia_organica_krig"]]
vars_scaled = StandardScaler().fit_transform(vars_cluster)

# ---------------------------------------------------------------------------
# 2. Método del codo para elegir k
# ---------------------------------------------------------------------------
inercias = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(vars_scaled)
    inercias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 7), inercias, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.savefig(OUT / "metodo_codo.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. KMeans con k=3 y etiquetas ordenadas por pH medio
# ---------------------------------------------------------------------------
km3 = KMeans(n_clusters=3, n_init=10, random_state=42)
cosecha["zona_manejo"] = km3.fit_predict(vars_scaled)

orden = cosecha.groupby("zona_manejo")["pH_krig"].mean().sort_values().index
mapa_orden = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
cosecha["zona_manejo"] = cosecha["zona_manejo"].map(mapa_orden)
etiquetas = {0: "Baja fertilidad", 1: "Media fertilidad", 2: "Alta fertilidad"}
cosecha["zona_manejo_label"] = cosecha["zona_manejo"].map(etiquetas)

print("Puntos por zona:")
print(cosecha["zona_manejo_label"].value_counts())
print("\nPromedios por zona:")
print(cosecha.groupby("zona_manejo_label")[
    ["pH_krig", "materia_organica_krig", "rendimiento_kg_ha"]].mean())

# ---------------------------------------------------------------------------
# 4. Mapa de zonas de manejo
# ---------------------------------------------------------------------------
gdf_zonas = gpd.GeoDataFrame(
    cosecha,
    geometry=gpd.points_from_xy(cosecha["lon"], cosecha["lat"]),
    crs="EPSG:4326",
)
colores = {"Baja fertilidad": "#d73027", "Media fertilidad": "#fee08b", "Alta fertilidad": "#1a9850"}

fig, ax = plt.subplots(figsize=(8, 7))
lote.boundary.plot(ax=ax, color="black", linewidth=1.5)
for zona, color in colores.items():
    gdf_zonas[gdf_zonas["zona_manejo_label"] == zona].plot(
        ax=ax, color=color, markersize=4, label=zona)
ax.set_title("Zonificación de manejo - lote Soldini")
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "zonificacion_manejo.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. ANOVA de una vía sobre el rendimiento + tamaño del efecto
# ---------------------------------------------------------------------------
grupos = [cosecha[cosecha["zona_manejo_label"] == z]["rendimiento_kg_ha"]
          for z in ["Baja fertilidad", "Media fertilidad", "Alta fertilidad"]]
f_stat, p_valor = stats.f_oneway(*grupos)
print(f"\nANOVA: F = {f_stat:.2f}, p-valor = {p_valor:.2e}")

gran_media = cosecha["rendimiento_kg_ha"].mean()
ss_between = sum(len(g) * (g.mean() - gran_media) ** 2 for g in grupos)
ss_total = ((cosecha["rendimiento_kg_ha"] - gran_media) ** 2).sum()
eta2 = ss_between / ss_total
print(f"Eta² (tamaño del efecto): {eta2:.3f}")

brecha = grupos[2].mean() - grupos[0].mean()
print(f"Brecha de rendimiento Alta vs Baja: {brecha:.0f} kg/ha")

# ---------------------------------------------------------------------------
# 6. Exportar
# ---------------------------------------------------------------------------
cosecha.to_csv(OUT / "dataset_final_dashboard.csv", index=False)
with open(OUT / "metricas_anova.json", "w") as f:
    json.dump({
        "f_stat": float(f_stat),
        "p_valor": float(p_valor),
        "eta2": float(eta2),
        "brecha_alta_baja_kg_ha": float(brecha),
    }, f, indent=2)

print("\nListo. Archivos guardados en outputs/")
