"""Dashboard e agente de previsão de demanda do Produto Alfa.

Rodar: streamlit run app/streamlit_app.py
Sem GEMINI_API_KEY o agente funciona em modo offline (roteamento por palavras-chave).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecastagent import data as D  # noqa: E402
from forecastagent import models as M  # noqa: E402
from forecastagent.tools import build_agent, t_drift  # noqa: E402

RED, INK, GREY = "#E3263B", "#1E1416", "#9A8A8D"
st.set_page_config(page_title="Previsão de demanda · Produto Alfa", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Rodando o backtest (8 janelas de 14 dias)…")
def cached_backtest():
    return M.backtest()


@st.cache_data
def cached_history():
    return D.load()


@st.cache_resource
def get_agent():
    return build_agent()


df = cached_history()
bt = cached_backtest()
agent = get_agent()
st.session_state.setdefault("pending", [])
st.session_state.setdefault("log", [])

st.title("Previsão de demanda do Produto Alfa")
st.caption("Agente de IA que prevê os próximos 14 dias, compara modelos e recomenda estoque. "
           "Os números vêm de código determinístico; o modelo de linguagem só interpreta e explica. "
           "Dados sintéticos com parâmetros conhecidos, para avaliar contra gabarito.")
if agent.llm is None:
    st.info("Modo offline: sem GEMINI_API_KEY, o agente escolhe ferramentas por palavras-chave.", icon="🔌")

tab_dash, tab_agent, tab_eval, tab_aprov = st.tabs(["Dashboard", "Agente", "Avaliação", "Aprovações"])

with tab_dash:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.subheader("Cenário")
        promo = st.multiselect("Dias com promoção (1 a 14)", list(range(1, 15)), default=[])
        estoque = st.number_input("Estoque atual (unidades)", 0, 10000, 300, step=50)
        servico = st.slider("Nível de serviço", 0.80, 0.99, 0.95, 0.01)
    fc = M.forecast(14, promo)
    with c1:
        hist = df.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["data"], y=hist["vendas"], name="Vendas reais", line=dict(color=GREY, width=1.5)))
        fig.add_trace(go.Scatter(x=list(fc["data"]) + list(fc["data"][::-1]), y=list(fc["p90"]) + list(fc["p10"][::-1]),
                                 fill="toself", fillcolor="rgba(227,38,59,0.15)", line=dict(width=0), name="Intervalo p10–p90"))
        fig.add_trace(go.Scatter(x=fc["data"], y=fc["previsao"], name=f"Previsão ({fc['modelo'].iloc[0]})",
                                 line=dict(color=RED, width=2.5)))
        promo_days = fc[fc["em_promocao"] == 1]
        if len(promo_days):
            fig.add_trace(go.Scatter(x=promo_days["data"], y=promo_days["previsao"], mode="markers", name="Promoção",
                                     marker=dict(color=INK, size=9, symbol="star")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15),
                          yaxis_title="unidades/dia")
        st.plotly_chart(fig, use_container_width=True)

    from forecastagent.tools import t_estoque

    rec = t_estoque(estoque, servico, promo)
    drift = t_drift()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Demanda prevista (14 dias)", f"{rec['demanda_prevista']:.0f} un")
    k2.metric("Pedido recomendado", f"{rec['pedido_recomendado']} un", help=rec["resumo"])
    melhor = min((r for r in bt["tabela"] if r["modelo"] != "naive_sazonal"), key=lambda r: r["mape"])
    k3.metric("MAPE do melhor modelo", f"{melhor['mape']:.2f}%", help=f"Modelo: {melhor['modelo']}")
    k4.metric("Drift (razão de MAE)", f"{drift['razao']:.2f}", delta="alerta" if drift["alerta"] else "estável",
              delta_color="inverse" if drift["alerta"] else "normal")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Backtest: 8 janelas de 14 dias")
        tab = pd.DataFrame(bt["tabela"]).rename(columns={"modelo": "Modelo", "mae": "MAE", "mape": "MAPE (%)", "vies": "Viés"})
        st.dataframe(tab, hide_index=True, use_container_width=True)
    with c4:
        st.subheader("O que move as vendas")
        drv = M.drivers()
        fat = pd.DataFrame({"Fator": ["Promoção", "Feriado nacional", "Tendência (ao ano)"],
                            "Efeito (%)": [drv["efeito_promocao_pct"], drv["efeito_feriado_pct"], drv["tendencia_pct_ano"]]})
        fig2 = go.Figure(go.Bar(x=fat["Efeito (%)"], y=fat["Fator"], orientation="h",
                                marker_color=[RED if v > 0 else GREY for v in fat["Efeito (%)"]]))
        fig2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)

with tab_agent:
    st.subheader("Pergunte ao agente")
    exemplos = ["Quanto vamos vender nos próximos 14 dias?", "Qual o efeito da promoção nas vendas?",
                "Quanto devo comprar se tenho estoque atual de 300 unidades?", "Compare os modelos.",
                "Publique a previsão para o time de compras."]
    ex = st.selectbox("Exemplos", [""] + exemplos)
    q = st.text_input("Pergunta", value=ex)
    if st.button("Perguntar", type="primary") and q.strip():
        with st.spinner("Consultando ferramentas…"):
            r = agent.run(q)
        st.session_state["log"].append(r)
        st.session_state["pending"].extend(r.pending)
    for r in reversed(st.session_state["log"][-5:]):
        with st.container(border=True):
            st.markdown(f"**{r.question}**")
            st.write(r.answer)
            badge = {"ok": "✅ números conferidos", "revisar": "⚠️ revisar: número sem fonte", "erro": "❌ erro"}[r.status]
            st.caption(f"{badge} · modo {r.mode} · ferramentas: {', '.join(c['tool'] for c in r.calls) or 'nenhuma'}")
            if r.ungrounded:
                st.warning(f"Números sem fonte nas ferramentas: {r.ungrounded}")
            if r.pending:
                st.info("Ação sensível aguardando aprovação na aba Aprovações.")
            with st.expander("Rastro (chamadas e resultados)"):
                st.json({"calls": r.calls, "results": r.results, "errors": r.errors}, expanded=False)

with tab_eval:
    st.subheader("Avaliação contra gabarito")
    res_path = ROOT / "eval" / "resultados.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        st.metric("Casos corretos", f"{res['acertos']} de {res['total']}")
        st.dataframe(pd.DataFrame(res["casos"])[["caso", "ok", "obtido", "esperado"]].astype(str),
                     hide_index=True, use_container_width=True)
    else:
        st.write("Rode `python eval/run_eval.py` para gerar os resultados.")

with tab_aprov:
    st.subheader("Ações que exigem aprovação humana")
    if not st.session_state["pending"]:
        st.write("Nenhuma ação pendente. Peça ao agente para publicar a previsão ou disparar um retreino.")
    for i, p in enumerate(list(st.session_state["pending"])):
        with st.container(border=True):
            st.markdown(f"**{p.tool}** · argumentos `{p.args}`")
            a, b = st.columns(2)
            if a.button("Aprovar e executar", key=f"ok{i}", type="primary"):
                out = agent.approve(p)
                st.session_state["pending"].remove(p)
                st.success(out["resumo"])
            if b.button("Recusar", key=f"no{i}"):
                st.session_state["pending"].remove(p)
                st.rerun()
