"""Modelos de previsão, backtest com origem móvel e explicação dos efeitos.

Todos os cálculos são determinísticos. O agente só chama estas funções.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import data as D

DOW = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]
MODELOS = ["naive_sazonal", "regressao_log", "gbm", "sarimax"]


def features(dates: pd.DatetimeIndex, promo: np.ndarray, feriado: np.ndarray, t0: pd.Timestamp) -> pd.DataFrame:
    t = (dates - t0).days.to_numpy(dtype=float)
    X = pd.DataFrame(index=dates)
    for k, name in enumerate(DOW[1:], start=1):  # segunda é a referência
        X[f"dow_{name}"] = (dates.dayofweek == k).astype(float)
    X["promocao"] = promo.astype(float)
    X["feriado"] = feriado.astype(float)
    X["t"] = t / 365.25
    doy = dates.dayofyear.to_numpy()
    for k in (1, 2):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return X


def holidays_for(dates: pd.DatetimeIndex) -> np.ndarray:
    br = holidays.country_holidays("BR", years=sorted(set(dates.year)))
    return np.array([d.date() in br for d in dates], dtype=int)


@dataclass
class Fitted:
    name: str
    model: object
    t0: pd.Timestamp
    history: pd.DataFrame

    def predict(self, dates: pd.DatetimeIndex, promo: np.ndarray, feriado: np.ndarray) -> np.ndarray:
        if self.name == "naive_sazonal":
            last7 = self.history["vendas"].to_numpy()[-7:]
            last_dates = self.history["data"].iloc[-7:]
            by_dow = {d.dayofweek: v for d, v in zip(last_dates, last7)}
            return np.array([by_dow[d.dayofweek] for d in dates], dtype=float)
        if self.name == "sarimax":
            exog = np.column_stack([promo, feriado]).astype(float)
            fc = self.model.forecast(steps=len(dates), exog=exog)
            return np.expm1(np.asarray(fc))
        X = features(dates, promo, feriado, self.t0)
        return np.expm1(self.model.predict(X))


def fit(name: str, hist: pd.DataFrame) -> Fitted:
    hist = hist.reset_index(drop=True)
    t0 = hist["data"].iloc[0]
    dates = pd.DatetimeIndex(hist["data"])
    y = np.log1p(hist["vendas"].to_numpy(dtype=float))
    if name == "naive_sazonal":
        return Fitted(name, None, t0, hist)
    if name == "sarimax":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = hist[["em_promocao", "feriado_nacional"]].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(y, exog=exog, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False, maxiter=60)
        return Fitted(name, m, t0, hist)
    X = features(dates, hist["em_promocao"].to_numpy(), hist["feriado_nacional"].to_numpy(), t0)
    if name == "regressao_log":
        m = Ridge(alpha=1e-3).fit(X, y)
    elif name == "gbm":
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
    else:
        raise ValueError(f"Modelo desconhecido: {name}. Use um de {MODELOS}.")
    return Fitted(name, m, t0, hist)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = y - p
    return {
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(y, 1)) * 100),
        "vies": float(np.mean(err)),
    }


@lru_cache(maxsize=8)
def backtest(folds: int = 8, horizon: int = 14, modelos: tuple[str, ...] = tuple(MODELOS)) -> dict:
    """Origem móvel: para cada uma das últimas `folds` janelas de `horizon` dias, treina com o
    passado e prevê a janela. Promoções e feriados futuros são conhecidos (plano comercial)."""
    df = D.load()
    n = len(df)
    rows, residuals = [], {m: [] for m in modelos}
    for k in range(folds, 0, -1):
        cut = n - k * horizon
        train, test = df.iloc[:cut], df.iloc[cut : cut + horizon]
        dates = pd.DatetimeIndex(test["data"])
        y = test["vendas"].to_numpy(dtype=float)
        for m in modelos:
            p = fit(m, train).predict(dates, test["em_promocao"].to_numpy(), test["feriado_nacional"].to_numpy())
            residuals[m].extend((y / np.maximum(p, 1)).tolist())
            rows.append({"modelo": m, "janela_inicio": str(dates[0].date()), **_metrics(y, p)})
    res = pd.DataFrame(rows)
    table = res.groupby("modelo")[["mae", "mape", "vies"]].mean().reindex(list(modelos)).round(2)
    return {"tabela": table.reset_index().to_dict("records"), "janelas": rows, "razoes": residuals}


def best_model(folds: int = 8) -> str:
    tab = backtest(folds)["tabela"]
    candidates = [r for r in tab if r["modelo"] != "naive_sazonal"]
    return min(candidates, key=lambda r: r["mape"])["modelo"]


def future_frame(horizon: int, promo_dias: list[int] | None) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    df = D.load()
    start = df["data"].iloc[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    promo = np.zeros(horizon, dtype=int)
    for d in promo_dias or []:
        if not 1 <= int(d) <= horizon:
            raise ValueError(f"dia de promoção {d} fora do horizonte 1..{horizon}")
        promo[int(d) - 1] = 1
    return dates, promo, holidays_for(dates)


def forecast(horizon: int = 14, promo_dias: list[int] | None = None, modelo: str | None = None) -> pd.DataFrame:
    if not 1 <= horizon <= 28:
        raise ValueError("horizonte deve estar entre 1 e 28 dias")
    modelo = modelo or best_model()
    df = D.load()
    dates, promo, fer = future_frame(horizon, promo_dias)
    p = fit(modelo, df).predict(dates, promo, fer)
    ratios = np.array(backtest()["razoes"][modelo])
    lo, hi = np.quantile(ratios, [0.1, 0.9])
    return pd.DataFrame({
        "data": dates, "previsao": np.round(np.maximum(p, 0), 1),
        "p10": np.round(np.maximum(p * lo, 0), 1), "p90": np.round(p * hi, 1),
        "em_promocao": promo, "feriado_nacional": fer, "modelo": modelo,
    })


def drivers() -> dict:
    """Ajusta a regressão em log em toda a série e devolve os efeitos multiplicativos."""
    df = D.load()
    m = fit("regressao_log", df).model
    coef = dict(zip(m.feature_names_in_, m.coef_))
    dow = {"seg": 0.0} | {k.split("_")[1]: v for k, v in coef.items() if k.startswith("dow_")}
    mean_dow = np.mean(list(dow.values()))
    return {
        "efeito_promocao_pct": round((np.exp(coef["promocao"]) - 1) * 100, 1),
        "efeito_feriado_pct": round((np.exp(coef["feriado"]) - 1) * 100, 1),
        "tendencia_pct_ano": round((np.exp(coef["t"]) - 1) * 100, 1),
        "fator_dia_semana": {k: round(float(np.exp(v - mean_dow)), 3) for k, v in dow.items()},
    }
