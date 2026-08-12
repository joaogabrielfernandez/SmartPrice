# Smart Price — Previsão de Valor de Aluguel Residencial

Plataforma em Streamlit que estima o valor ideal de aluguel de um imóvel residencial a partir de suas
características e da localização, com navegação por cidade e bairro, mapa interativo e simulador.

**App público:** _(cole aqui o link do Streamlit Cloud depois do deploy)_

---

## O que tem aqui

| arquivo | o que é |
|---|---|
| `app.py` | aplicação Streamlit (filtros, gráficos, mapa, simulador) |
| `model.joblib` | LightGBM final + todo o pré-processamento (encoders, tabelas de imputação, métricas) |
| `dados_app.parquet` | base modelável enxuta (21.650 anúncios) usada nos gráficos e filtros |
| `ref_bairro.parquet` | referência por bairro: mediana de aluguel, R$/m², condomínio/m², coordenadas |
| `treinar_modelo.py` | script que regenera os três artefatos acima a partir do CSV bruto |
| `smart_price.ipynb` | notebook com EDA, decisões de limpeza, comparação de modelos e diagnóstico |
| `requirements.txt` | dependências com versão fixada |

A base bruta (`dataZAP.csv`, 48 MB) **não vai versionada** — está no `.gitignore`. Os artefatos já
vêm prontos no repositório, então o app sobe sem precisar dela.

## Resultados do modelo

LightGBM sobre o log do aluguel, escolhido em comparação com Regressão Linear, Árvore de Decisão,
Random Forest e XGBoost no mesmo split.

| métrica | valor |
|---|---|
| MAE | R$ 1.049 |
| RMSE | R$ 2.405 |
| MAPE | 22,3% |
| R² | 0,822 |
| previsões dentro de ±20% | 57,9% |
| previsões dentro de ±30% | 75,8% |

Cobertura: 56 cidades, 2.624 bairros, 9 estados.

## Rodar localmente

```bash
git clone https://github.com/SEU_USUARIO/smart-price.git
cd smart-price
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar os artefatos do zero (opcional — baixe o CSV do
[Kaggle](https://www.kaggle.com/datasets/maverickjpa/brazilian-real-estate-to-rent/data)):

```bash
python treinar_modelo.py dataZAP.csv
```

## Deploy no Streamlit Community Cloud

1. Suba este repositório no GitHub (público).
2. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com a conta do GitHub.
3. **Create app → Deploy a public app from GitHub**, escolha o repositório, branch `main`,
   arquivo principal `app.py`.
4. **Deploy**. Na primeira subida o Streamlit instala o `requirements.txt` (leva alguns minutos).
5. Copie o link gerado e cole na seção "App público" acima.

## Limites conhecidos

- A base é de julho de 2020 — aplique reajuste por índice antes de usar como valor de contrato.
- Acima de R$ 6 mil o modelo subestima em média; nessa faixa a sugestão é **piso de negociação**.
- Cobertura é o tipo menos previsível (MAPE ~28%); flat é o mais previsível (~16%).
- Estado de conservação, andar, vista e idade do prédio não existem na base e respondem por boa
  parte do resíduo.

## Fonte dos dados

[Brazilian Real Estate to Rent — Kaggle](https://www.kaggle.com/datasets/maverickjpa/brazilian-real-estate-to-rent/data)
(anúncios do ZAP Imóveis coletados até julho de 2020).
