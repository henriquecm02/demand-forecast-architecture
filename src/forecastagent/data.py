"""Gerador da série sintética de vendas diárias do Produto Alfa.

O desafio da Valcann não traz dados. Para que o agente possa ser avaliado contra um
gabarito, a série é gerada a partir de parâmetros conhecidos (gravados em
data/parametros_reais.json): tendência, sazonalidade semanal e anual, feriados
nacionais e o efeito das promoções. O teste confere se o modelo recupera esses efeitos.
"""
from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CSV = DATA_DIR / "vendas_produto_alfa.csv"
PARAMS = DATA_DIR / "parametros_reais.json"

TRUE_PARAMS = {
    "inicio": "2022-01-01",
    "fim": "2024-12-31",
    "nivel_base": 120.0,
    "tendencia_por_dia": 0.03,
    "fator_dia_semana": {"seg": 0.90, "ter": 0.95, "qua": 1.00, "qui": 1.00, "sex": 1.15, "sab": 1.30, "dom": 0.80},
    "amplitude_anual": 0.08,
    "efeito_feriado": -0.25,
    "efeito_promocao": 0.35,
    "ruido_desvio_relativo": 0.06,
    "semente": 42,
}


def generate(params: dict = TRUE_PARAMS) -> pd.DataFrame:
    rng = np.random.default_rng(params["semente"])
    dates = pd.date_range(params["inicio"], params["fim"], freq="D")
    n = len(dates)
    t = np.arange(n)
    br = holidays.country_holidays("BR", years=sorted(set(dates.year)))
    feriado = np.array([d.date() in br for d in dates], dtype=int)

    # promoções em blocos de 3 a 5 dias, cerca de uma por mês
    promo = np.zeros(n, dtype=int)
    i = int(rng.integers(5, 20))
    while i < n:
        dur = int(rng.integers(3, 6))
        promo[i : i + dur] = 1
        i += dur + int(rng.integers(20, 40))

    dow = np.array(list(params["fator_dia_semana"].values()))[dates.dayofweek]
    anual = 1 + params["amplitude_anual"] * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
    nivel = params["nivel_base"] + params["tendencia_por_dia"] * t
    media = nivel * dow * anual * (1 + params["efeito_feriado"] * feriado) * (1 + params["efeito_promocao"] * promo)
    ruido = rng.normal(0, params["ruido_desvio_relativo"], n)
    vendas = np.maximum(0, np.round(media * (1 + ruido))).astype(int)
    return pd.DataFrame({"data": dates, "vendas": vendas, "em_promocao": promo, "feriado_nacional": feriado})


def write() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(CSV, index=False)
    PARAMS.write_text(json.dumps(TRUE_PARAMS, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def load() -> pd.DataFrame:
    if not CSV.exists():
        return write()
    return pd.read_csv(CSV, parse_dates=["data"])


if __name__ == "__main__":
    d = write()
    print(f"{len(d)} dias gravados em {CSV}")
