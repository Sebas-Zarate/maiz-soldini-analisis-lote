"""
Paso 2 - Interpolación espacial del suelo por kriging ordinario
Proyecto: Diagnóstico de lote de maíz - Soldini, Santa Fe

Qué hace:
  - Interpola pH y materia orgánica desde 65 muestras puntuales a todo el lote
    (kriging ordinario con variograma esférico, PyKrige)
  - Enmascara la grilla con el polígono real del lote
  - Asigna a cada punto del monitor de cosecha su pH y materia orgánica interpolados
  - Chequea la correlación de cada variable con el rendimiento

Entradas (data/):    cosecha_monitor.csv, muestreo_suelo.csv, lote_soldini.geojson
Salidas  (outputs/): cosecha_con_suelo_interpolado.csv, kriging_grids.npz,
                     kriging_ph.png, kriging_materia_organica.png

Uso:  python scripts/02_kriging.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from pykrige.ok import OrdinaryKriging

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

cosecha = pd.read_csv(DATA / "cosecha_monitor.csv")
suelo = pd.read_csv(DATA / "muestreo_suelo.csv")
lote = gpd.read_file(DATA / "lote_soldini.geojson")
poligono = lote.geometry.iloc[0]


# ---------------------------------------------------------------------------
# 1. Grilla regular sobre el bounding box del lote (en grados)
# ---------------------------------------------------------------------------
minx, miny, maxx, maxy = lote.total_bounds
grid_lon = np.linspace(minx, maxx, 200)
grid_lat = np.linspace(miny, maxy, 200)

# Máscara: qué puntos de la grilla caen dentro del polígono real
gx, gy = np.meshgrid(grid_lon, grid_lat)
mask = shapely.contains_xy(poligono, gx, gy)


# ---------------------------------------------------------------------------
# 2. Kriging ordinario de pH y materia orgánica
# ---------------------------------------------------------------------------
def krigear(variable):
    ok = OrdinaryKriging(
        suelo["lon"], suelo["lat"], suelo[variable],
        variogram_model="spherical",
        verbose=False, enable_plotting=False,
    )
    z, _ = ok.execute("grid", grid_lon, grid_lat)
    return ok, z


ok_ph, z_ph = krigear("pH")
ok_mo, z_mo = krigear("materia_organica_pct")
print(f"Rango interpolado de pH: {z_ph.min():.2f} - {z_ph.max():.2f}")
print(f"Rango interpolado de materia orgánica: {z_mo.min():.2f} - {z_mo.max():.2f} %")

z_ph_masked = np.ma.masked_where(~mask, z_ph)
z_mo_masked = np.ma.masked_where(~mask, z_mo)


# ---------------------------------------------------------------------------
# 3. Mapas de suelo interpolado
# ---------------------------------------------------------------------------
def mapa(z_masked, cmap, etiqueta, titulo, archivo):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(grid_lon, grid_lat, z_masked, cmap=cmap, shading="auto")
    lote.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.scatter(suelo["lon"], suelo["lat"], c="red", s=15,
               edgecolor="white", linewidth=0.5, label="Muestras de suelo")
    plt.colorbar(im, ax=ax, label=etiqueta)
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / archivo, dpi=200, bbox_inches="tight")
    plt.close()


mapa(z_ph_masked, "viridis", "pH", "pH interpolado (kriging) - lote Soldini", "kriging_ph.png")
mapa(z_mo_masked, "YlGn", "Materia orgánica (%)",
     "Materia orgánica interpolada (kriging) - lote Soldini", "kriging_materia_organica.png")


# ---------------------------------------------------------------------------
# 4. Interpolar directamente en las coordenadas de cada punto de cosecha
# ---------------------------------------------------------------------------
ph_en_cosecha, _ = ok_ph.execute("points", cosecha["lon"].values, cosecha["lat"].values)
mo_en_cosecha, _ = ok_mo.execute("points", cosecha["lon"].values, cosecha["lat"].values)

cosecha["pH_krig"] = ph_en_cosecha
cosecha["materia_organica_krig"] = mo_en_cosecha

print("\nCorrelación rendimiento vs pH interpolado:",
      round(cosecha["rendimiento_kg_ha"].corr(cosecha["pH_krig"]), 3))
print("Correlación rendimiento vs materia orgánica interpolada:",
      round(cosecha["rendimiento_kg_ha"].corr(cosecha["materia_organica_krig"]), 3))


# ---------------------------------------------------------------------------
# 5. Exportar para los pasos siguientes y para el dashboard
# ---------------------------------------------------------------------------
cosecha.to_csv(OUT / "cosecha_con_suelo_interpolado.csv", index=False)
np.savez(
    OUT / "kriging_grids.npz",
    grid_lon=grid_lon, grid_lat=grid_lat,
    z_ph=z_ph_masked.filled(np.nan),
    z_mo=z_mo_masked.filled(np.nan),
)
print("\nListo. Archivos guardados en outputs/")
