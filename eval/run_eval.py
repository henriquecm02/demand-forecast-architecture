"""Avaliação contra gabarito. Roda sem GEMINI_API_KEY.

Três blocos:
1. Os modelos recuperam os parâmetros reais da série sintética?
2. O agente escolhe a ferramenta certa (modo offline) e segura ações sensíveis?
3. A conferência de números barra respostas com números inventados?

Uso: python eval/run_eval.py [--gemini]  (grava eval/resultados[_gemini].json e .md)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecastagent import data as D  # noqa: E402
from forecastagent import models as M  # noqa: E402
from forecastagent.agentcore import ScriptedLLM  # noqa: E402
from forecastagent.tools import build_agent  # noqa: E402

ROTEAMENTO = [
    ("Quanto vamos vender nos próximos 14 dias?", "prever_demanda"),
    ("Qual a previsão de demanda com promoção nos dias 3, 4 e 5?", "prever_demanda"),
    ("Qual modelo erra menos? Compare os modelos.", "comparar_modelos"),
    ("Qual o MAPE do melhor modelo no backtest?", "comparar_modelos"),
    ("Qual o efeito da promoção nas vendas?", "explicar_fatores"),
    ("Por que o sábado vende mais?", "explicar_fatores"),
    ("Quanto devo comprar se tenho estoque atual de 300 unidades?", "recomendar_estoque"),
    ("O modelo piorou nas últimas semanas? Tem drift?", "verificar_drift"),
    ("Publique a previsão para o time de compras.", "publicar_previsao"),
    ("Dispare o retreino do modelo.", "disparar_retreino"),
]


def check(nome: str, ok: bool, obtido, esperado) -> dict:
    return {"caso": nome, "ok": bool(ok), "obtido": obtido, "esperado": esperado}


def main() -> int:
    real = json.loads(D.PARAMS.read_text(encoding="utf-8")) if D.PARAMS.exists() else D.TRUE_PARAMS
    out: list[dict] = []

    # 1. modelos contra os parâmetros reais
    drv = M.drivers()
    promo_real = real["efeito_promocao"] * 100
    fer_real = real["efeito_feriado"] * 100
    out.append(check("efeito da promoção recuperado (±5 p.p.)", abs(drv["efeito_promocao_pct"] - promo_real) <= 5,
                     float(drv["efeito_promocao_pct"]), promo_real))
    out.append(check("efeito do feriado recuperado (±5 p.p.)", abs(drv["efeito_feriado_pct"] - fer_real) <= 5,
                     float(drv["efeito_feriado_pct"]), fer_real))
    f = drv["fator_dia_semana"]
    razao_real = real["fator_dia_semana"]["sab"] / real["fator_dia_semana"]["seg"]
    out.append(check("sábado/segunda recuperado (±3%)", abs(f["sab"] / f["seg"] / razao_real - 1) <= 0.03,
                     round(f["sab"] / f["seg"], 3), round(razao_real, 3)))
    tab = {r["modelo"]: r for r in M.backtest()["tabela"]}
    melhor = M.best_model()
    out.append(check("melhor modelo tem metade do MAPE do naive sazonal",
                     tab[melhor]["mape"] <= 0.5 * tab["naive_sazonal"]["mape"],
                     {melhor: tab[melhor]["mape"]}, {"naive_sazonal": tab["naive_sazonal"]["mape"]}))
    fc = M.forecast(14, [3, 4, 5])
    out.append(check("previsão: 14 dias, sem negativos, p10 <= previsão <= p90",
                     len(fc) == 14 and (fc["previsao"] >= 0).all() and (fc["p10"] <= fc["previsao"]).all()
                     and (fc["previsao"] <= fc["p90"]).all(), len(fc), 14))
    promo_lift = fc.loc[fc["em_promocao"] == 1, "previsao"].mean() / fc.loc[fc["em_promocao"] == 0, "previsao"].mean()
    out.append(check("dias com promoção previstos acima dos demais", promo_lift > 1.15, round(float(promo_lift), 2), "> 1.15"))

    # 2. roteamento e aprovação (offline por padrão; --gemini usa o modelo real)
    usar_gemini = "--gemini" in sys.argv
    agent = build_agent(use_env=usar_gemini)
    if usar_gemini and agent.llm is None:
        print("--gemini pedido, mas GEMINI_API_KEY não está definida")
        return 2
    for pergunta, esperado in ROTEAMENTO:
        r = agent.run(pergunta)
        tools = [c["tool"] for c in r.calls]
        ok = tools[:1] == [esperado]
        if esperado in ("publicar_previsao", "disparar_retreino"):
            ok = ok and r.pending and not r.results  # nada roda sem aprovação
        else:
            ok = ok and r.status == "ok"
        out.append(check(f"roteamento: {pergunta}", ok, tools, esperado))

    # 3. conferência de números com um LLM simulado que inventa um valor
    llm = ScriptedLLM([
        {"calls": [{"tool": "explicar_fatores", "args_json": "{}"}]},
        {"answer": "A promoção aumenta as vendas em 52%.", "numbers_cited": [52]},
    ])
    r = build_agent(llm=llm, use_env=False).run("Qual o efeito da promoção?")
    out.append(check("número inventado é barrado (status revisar)", r.status == "revisar" and 52.0 in r.ungrounded,
                     r.status, "revisar"))
    llm = ScriptedLLM([
        {"calls": [{"tool": "explicar_fatores", "args_json": "{}"}]},
        {"answer": f"A promoção aumenta as vendas em {drv['efeito_promocao_pct']:.1f}%.",
         "numbers_cited": [float(drv["efeito_promocao_pct"])]},
    ])
    r = build_agent(llm=llm, use_env=False).run("Qual o efeito da promoção?")
    out.append(check("número correto passa (status ok)", r.status == "ok", r.status, "ok"))

    acertos = sum(c["ok"] for c in out)
    resumo = {"acertos": acertos, "total": len(out), "backtest": M.backtest()["tabela"], "fatores": {
        k: (float(v) if not isinstance(v, dict) else v) for k, v in drv.items()}, "casos": out}
    sufixo = "_gemini" if usar_gemini else ""
    (ROOT / "eval" / f"resultados{sufixo}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    md = [f"# Resultados da avaliação\n\n**{acertos} de {len(out)} casos corretos.**\n",
          "| Caso | Resultado | Obtido | Esperado |", "|---|---|---|---|"]
    md += [f"| {c['caso']} | {'ok' if c['ok'] else 'FALHOU'} | {c['obtido']} | {c['esperado']} |" for c in out]
    md += ["\n## Backtest (8 janelas de 14 dias)\n", "| Modelo | MAE | MAPE (%) | Viés |", "|---|---|---|---|"]
    md += [f"| {r['modelo']} | {r['mae']:.2f} | {r['mape']:.2f} | {r['vies']:.2f} |" for r in M.backtest()["tabela"]]
    (ROOT / "eval" / f"resultados{sufixo}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{acertos}/{len(out)} casos corretos")
    for c in out:
        if not c["ok"]:
            print("FALHOU:", c)
    return 0 if acertos == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
