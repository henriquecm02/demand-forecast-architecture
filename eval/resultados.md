# Resultados da avaliação

**18 de 18 casos corretos.**

| Caso | Resultado | Obtido | Esperado |
|---|---|---|---|
| efeito da promoção recuperado (±5 p.p.) | ok | 35.5 | 35.0 |
| efeito do feriado recuperado (±5 p.p.) | ok | -23.7 | -25.0 |
| sábado/segunda recuperado (±3%) | ok | 1.435 | 1.444 |
| melhor modelo tem metade do MAPE do naive sazonal | ok | {'regressao_log': 4.42} | {'naive_sazonal': 17.05} |
| previsão: 14 dias, sem negativos, p10 <= previsão <= p90 | ok | 14 | 14 |
| dias com promoção previstos acima dos demais | ok | 1.51 | > 1.15 |
| roteamento: Quanto vamos vender nos próximos 14 dias? | ok | ['prever_demanda'] | prever_demanda |
| roteamento: Qual a previsão de demanda com promoção nos dias 3, 4 e 5? | ok | ['prever_demanda'] | prever_demanda |
| roteamento: Qual modelo erra menos? Compare os modelos. | ok | ['comparar_modelos'] | comparar_modelos |
| roteamento: Qual o MAPE do melhor modelo no backtest? | ok | ['comparar_modelos'] | comparar_modelos |
| roteamento: Qual o efeito da promoção nas vendas? | ok | ['explicar_fatores'] | explicar_fatores |
| roteamento: Por que o sábado vende mais? | ok | ['explicar_fatores'] | explicar_fatores |
| roteamento: Quanto devo comprar se tenho estoque atual de 300 unidades? | ok | ['recomendar_estoque'] | recomendar_estoque |
| roteamento: O modelo piorou nas últimas semanas? Tem drift? | ok | ['verificar_drift'] | verificar_drift |
| roteamento: Publique a previsão para o time de compras. | ok | ['publicar_previsao'] | publicar_previsao |
| roteamento: Dispare o retreino do modelo. | ok | ['disparar_retreino'] | disparar_retreino |
| número inventado é barrado (status revisar) | ok | revisar | revisar |
| número correto passa (status ok) | ok | ok | ok |

## Backtest (8 janelas de 14 dias)

| Modelo | MAE | MAPE (%) | Viés |
|---|---|---|---|
| naive_sazonal | 26.80 | 17.05 | -0.77 |
| regressao_log | 6.76 | 4.42 | 0.40 |
| gbm | 9.28 | 5.95 | 1.58 |
| sarimax | 7.09 | 4.71 | -0.96 |
