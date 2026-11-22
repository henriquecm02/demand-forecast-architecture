"""Ferramentas que o agente pode chamar. Cada uma devolve um dict com um campo 'resumo'."""
from __future__ import annotations

import json
import math
import unicodedata
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np

from . import data as D
from . import models as M
from .agentcore import Agent, Tool, make_llm

REPORTS = D.ROOT / "reports"


def _f(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def t_prever(horizonte: int = 14, dias_promocao: list[int] | None = None) -> dict:
    fc = M.forecast(int(horizonte), dias_promocao)
    total = float(fc["previsao"].sum())
    pico = fc.loc[fc["previsao"].idxmax()]
    return {
        "modelo": str(fc["modelo"].iloc[0]),
        "total_previsto": _f(total),
        "media_diaria": _f(total / len(fc)),
        "dia_pico": str(pico["data"].date()),
        "venda_pico": _f(pico["previsao"]),
        "dias": [
            {"data": str(r.data.date()), "previsao": _f(r.previsao), "p10": _f(r.p10), "p90": _f(r.p90),
             "promocao": int(r.em_promocao), "feriado": int(r.feriado_nacional)}
            for r in fc.itertuples()
        ],
        "resumo": (f"Previsão de {len(fc)} dias com o modelo {fc['modelo'].iloc[0]}: total de {total:.1f} unidades "
                   f"(média de {total / len(fc):.1f} por dia). Pico em {pico['data'].date()} com {pico['previsao']:.1f} unidades."),
    }


def t_backtest() -> dict:
    tab = M.backtest()["tabela"]
    melhor = min((r for r in tab if r["modelo"] != "naive_sazonal"), key=lambda r: r["mape"])
    base = next(r for r in tab if r["modelo"] == "naive_sazonal")
    linhas = "; ".join(f"{r['modelo']}: MAE {r['mae']:.2f}, MAPE {r['mape']:.2f}%" for r in tab)
    return {
        "tabela": tab, "melhor_modelo": melhor["modelo"], "janelas": 8, "horizonte": 14,
        "resumo": (f"Backtest com 8 janelas de 14 dias. {linhas}. Melhor modelo: {melhor['modelo']} "
                   f"(MAPE {melhor['mape']:.2f}% contra {base['mape']:.2f}% do naive sazonal)."),
    }


def t_explicar() -> dict:
    d = M.drivers()
    d = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    sab, dom = d["fator_dia_semana"]["sab"], d["fator_dia_semana"]["dom"]
    return {
        **d,
        "resumo": (f"Promoção aumenta as vendas em {d['efeito_promocao_pct']:.1f}%; feriado nacional muda em "
                   f"{d['efeito_feriado_pct']:.1f}%; a tendência é de {d['tendencia_pct_ano']:.1f}% ao ano. "
                   f"Sábado vende {sab:.3f}x a média da semana e domingo {dom:.3f}x."),
    }


def t_estoque(estoque_atual: float = 0, nivel_servico: float = 0.95, dias_promocao: list[int] | None = None,
              horizonte: int = 14) -> dict:
    if not 0.5 <= float(nivel_servico) < 1:
        raise ValueError("nivel_servico deve estar entre 0,5 e 0,99")
    fc = M.forecast(int(horizonte), dias_promocao)
    modelo = fc["modelo"].iloc[0]
    janelas = [w for w in M.backtest()["janelas"] if w["modelo"] == modelo]
    mae = float(np.mean([w["mae"] for w in janelas]))
    sd_dia = mae * 1.25  # MAE -> desvio padrão sob erro aproximadamente normal
    demanda = float(fc["previsao"].sum())
    z = NormalDist().inv_cdf(float(nivel_servico))
    seguranca = z * sd_dia * math.sqrt(len(fc))
    pedido = max(0, math.ceil(demanda + seguranca - float(estoque_atual)))
    return {
        "demanda_prevista": _f(demanda), "estoque_seguranca": _f(seguranca), "estoque_atual": _f(estoque_atual),
        "nivel_servico": float(nivel_servico), "pedido_recomendado": pedido,
        "resumo": (f"Demanda prevista de {demanda:.1f} unidades em {len(fc)} dias. Com nível de serviço de "
                   f"{float(nivel_servico) * 100:.0f}% o estoque de segurança é {seguranca:.1f}; descontando o estoque "
                   f"atual de {float(estoque_atual):.0f}, o pedido recomendado é de {pedido} unidades."),
    }


def t_drift(janela: int = 28) -> dict:
    df = D.load()
    modelo = M.best_model()
    train, test = df.iloc[:-janela], df.iloc[-janela:]
    import pandas as pd

    p = M.fit(modelo, train).predict(pd.DatetimeIndex(test["data"]), test["em_promocao"].to_numpy(),
                                     test["feriado_nacional"].to_numpy())
    mae_recente = float(np.mean(np.abs(test["vendas"].to_numpy() - p)))
    mae_ref = float(next(r["mae"] for r in M.backtest()["tabela"] if r["modelo"] == modelo))
    razao = mae_recente / mae_ref
    alerta = razao > 1.3
    return {
        "modelo": modelo, "mae_recente": _f(mae_recente, 2), "mae_backtest": _f(mae_ref, 2), "razao": _f(razao, 2),
        "alerta": alerta,
        "resumo": (f"Nos últimos {janela} dias o MAE foi {mae_recente:.2f} contra {mae_ref:.2f} no backtest "
                   f"(razão {razao:.2f}). " + ("Sinal de drift: vale retreinar." if alerta else "Sem sinal de drift.")),
    }


def t_publicar(horizonte: int = 14, dias_promocao: list[int] | None = None) -> dict:
    """Ação sensível: grava a previsão oficial usada pelo time de compras."""
    fc = M.forecast(int(horizonte), dias_promocao)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "previsao_publicada.csv"
    fc.to_csv(path, index=False)
    total = float(fc["previsao"].sum())
    return {"arquivo": str(path.relative_to(D.ROOT)), "total_previsto": _f(total),
            "resumo": f"Previsão de {len(fc)} dias publicada em {path.name} (total de {total:.1f} unidades)."}


def t_retreinar(motivo: str = "") -> dict:
    """Ação sensível: registra o pedido de retreino (no ambiente real dispararia o pipeline)."""
    REPORTS.mkdir(exist_ok=True)
    log = REPORTS / "retreinos.jsonl"
    M.backtest.cache_clear()
    entry = {"quando": datetime.now(timezone.utc).isoformat(timespec="seconds"), "motivo": motivo,
             "melhor_modelo": M.best_model()}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {**entry, "resumo": f"Retreino registrado. Modelo escolhido: {entry['melhor_modelo']}."}


TOOLS = [
    Tool("prever_demanda", "Prevê vendas diárias com intervalo p10-p90.",
         {"horizonte": "int, 1 a 28, padrão 14", "dias_promocao": "lista de int, dias do horizonte com promoção"}, t_prever),
    Tool("comparar_modelos", "Backtest com origem móvel: MAE e MAPE de naive sazonal, regressão, GBM e SARIMAX.", {}, t_backtest),
    Tool("explicar_fatores", "Efeito de promoção, feriado, tendência e dia da semana.", {}, t_explicar),
    Tool("recomendar_estoque", "Pedido de compra para cobrir o horizonte com um nível de serviço.",
         {"estoque_atual": "número, padrão 0", "nivel_servico": "0,5 a 0,99, padrão 0,95",
          "dias_promocao": "lista de int", "horizonte": "int, padrão 14"}, t_estoque),
    Tool("verificar_drift", "Compara o erro recente do modelo com o erro do backtest.", {"janela": "int, padrão 28"}, t_drift),
    Tool("publicar_previsao", "Grava a previsão oficial para o time de compras.",
         {"horizonte": "int", "dias_promocao": "lista de int"}, t_publicar, sensitive=True),
    Tool("disparar_retreino", "Registra o retreino do modelo.", {"motivo": "texto"}, t_retreinar, sensitive=True),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


def offline_router(question: str) -> list[dict]:
    """Roteamento por palavras-chave para o modo sem GEMINI_API_KEY."""
    q = _norm(question)
    import re

    dias = [int(x) for x in re.findall(r"dias? (\d+(?:\s*(?:,|e)\s*\d+)*)", q) for x in re.findall(r"\d+", x)] if "promo" in q else []
    estoque = re.search(r"estoque (?:atual )?(?:de )?(\d+)", q)
    calls = []
    if any(w in q for w in ("publi", "oficializ")):
        calls.append({"tool": "publicar_previsao", "args": {"dias_promocao": dias}})
    elif "retrein" in q:
        calls.append({"tool": "disparar_retreino", "args": {"motivo": question}})
    elif any(w in q for w in ("estoque", "comprar", "pedido")):
        calls.append({"tool": "recomendar_estoque",
                      "args": {"estoque_atual": int(estoque.group(1)) if estoque else 0, "dias_promocao": dias}})
    elif any(w in q for w in ("compar", "backtest", "melhor modelo", "erro", "mape")):
        calls.append({"tool": "comparar_modelos", "args": {}})
    elif any(w in q for w in ("drift", "degrad", "piorou")):
        calls.append({"tool": "verificar_drift", "args": {}})
    elif any(w in q for w in ("por que", "fator", "efeito", "impacto", "explica")):
        calls.append({"tool": "explicar_fatores", "args": {}})
    elif any(w in q for w in ("prev", "proxim", "demanda", "vender", "vendas")):
        calls.append({"tool": "prever_demanda", "args": {"dias_promocao": dias}})
    return calls


def build_agent(llm=None, use_env: bool = True) -> Agent:
    if llm is None and use_env:
        llm = make_llm()
    return Agent(TOOLS, llm=llm, offline_router=offline_router)
