"""
Paso 3 - Balance hídrico del ciclo con foco en el período crítico
Proyecto: Diagnóstico de lote de maíz - Soldini, Santa Fe

Qué hace:
  - Calcula el balance hídrico semanal: precipitación - evapotranspiración (ET)
  - Usa una ET simplificada: 34 mm/semana en el período crítico (semanas 10-16,
    floración-llenado de grano) y 28 mm/semana en el resto del ciclo
  - Acumula el balance a lo largo de la campaña
  - Calcula el déficit hídrico acumulado durante el período crítico

Entradas (data/):    clima_semanal.csv
Salidas  (outputs/): clima_balance.csv, balance_hidrico.png

Nota: es un balance simplificado (ET fija por etapa, sin almacenaje de agua en el
suelo ni coeficientes de cultivo). Sirve para dimensionar el estrés hídrico relativo
del ciclo, no para reemplazar un balance hídrico agronómico completo.

Uso:  python scripts/03_balance_hidrico.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Supuestos del balance
ET_CRITICA = 34.0   # mm/semana, floración-llenado de grano
ET_RESTO = 28.0     # mm/semana, resto del ciclo (aproximado)
SEMANA_INI_CRITICA, SEMANA_FIN_CRITICA = 10, 16

clima = pd.read_csv(DATA / "clima_semanal.csv", parse_dates=["fecha_inicio", "fecha_fin"])

# ---------------------------------------------------------------------------
# 1. Balance semanal y acumulado
# ---------------------------------------------------------------------------
en_critico = clima["semana"].between(SEMANA_INI_CRITICA, SEMANA_FIN_CRITICA)
clima["et_semanal"] = np.where(en_critico, ET_CRITICA, ET_RESTO)
clima["balance_semanal"] = clima["precipitacion_mm"] - clima["et_semanal"]
clima["balance_acumulado"] = clima["balance_semanal"].cumsum()

# ---------------------------------------------------------------------------
# 2. Déficit hídrico en el período crítico
#    (suma semanal de ET - precipitación, contando solo las semanas con déficit)
# ---------------------------------------------------------------------------
periodo_critico = clima[en_critico]
deficit_total_mm = (ET_CRITICA - periodo_critico["precipitacion_mm"]).clip(lower=0).sum()
print(f"Déficit hídrico acumulado en período crítico: {deficit_total_mm:.1f} mm")
print(f"Balance acumulado al final del ciclo: {clima['balance_acumulado'].iloc[-1]:.1f} mm")

# ---------------------------------------------------------------------------
# 3. Gráfico
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(clima["semana"], clima["balance_acumulado"], marker="o")
plt.axvspan(SEMANA_INI_CRITICA, SEMANA_FIN_CRITICA, color="orange", alpha=0.2,
            label="Período crítico (floración-llenado)")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Semana")
plt.ylabel("Balance hídrico acumulado (mm)")
plt.title("Balance hídrico acumulado - ciclo maíz Soldini 2025/26")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "balance_hidrico.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. Exportar
# ---------------------------------------------------------------------------
clima.to_csv(OUT / "clima_balance.csv", index=False)
print("\nListo. Archivos guardados en outputs/")
