"""Testes sem rede: ferramentas determinísticas, agente em modo offline e com LLM simulado."""
from forecastagent import models as M
from forecastagent.agentcore import ScriptedLLM, numbers_in_text, ungrounded_numbers
from forecastagent.tools import build_agent, t_estoque, t_prever


def test_forecast_shape_and_bounds():
    fc = M.forecast(14, [2, 3])
    assert len(fc) == 14
    assert (fc["previsao"] >= 0).all()
    assert (fc["p10"] <= fc["previsao"]).all() and (fc["previsao"] <= fc["p90"]).all()


def test_promo_out_of_horizon_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        M.forecast(14, [20])


def test_drivers_recover_true_promo_effect():
    assert abs(M.drivers()["efeito_promocao_pct"] - 35) <= 5


def test_stock_recommendation_discounts_current_stock():
    a = t_estoque(estoque_atual=0)
    b = t_estoque(estoque_atual=500)
    assert a["pedido_recomendado"] - b["pedido_recomendado"] in (499, 500, 501)


def test_sensitive_action_is_not_executed_without_approval(tmp_path):
    r = build_agent(use_env=False).run("Publique a previsão para o time de compras.")
    assert r.pending and not r.results


def test_ungrounded_number_is_flagged():
    llm = ScriptedLLM([
        {"calls": [{"tool": "prever_demanda", "args_json": "{}"}]},
        {"answer": "Vamos vender 99999 unidades.", "numbers_cited": [99999]},
    ])
    r = build_agent(llm=llm, use_env=False).run("Quanto vamos vender?")
    assert r.status == "revisar" and 99999.0 in r.ungrounded


def test_grounded_answer_passes():
    total = t_prever()["total_previsto"]
    llm = ScriptedLLM([
        {"calls": [{"tool": "prever_demanda", "args_json": "{\"horizonte\": 14}"}]},
        {"answer": f"O total previsto é {total:.1f} unidades.", "numbers_cited": [total]},
    ])
    r = build_agent(llm=llm, use_env=False).run("Quanto vamos vender nos próximos 14 dias?")
    assert r.status == "ok", r.ungrounded


def test_unknown_tool_is_reported():
    llm = ScriptedLLM([{"calls": [{"tool": "apagar_tudo", "args_json": "{}"}]}])
    r = build_agent(llm=llm, use_env=False).run("x")
    assert r.status == "erro"


def test_number_parsing_ptbr():
    assert numbers_in_text("1.234,5 e 12,5% e 2.000") == [1234.5, 12.5, 2000.0]
    assert ungrounded_numbers([35.0], {"x": 0.35}) == []
