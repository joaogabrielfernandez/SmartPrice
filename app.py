"""
Smart Price - plataforma de precificacao de aluguel residencial
Streamlit app. Consome os artefatos gerados por treinar_modelo.py.
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Smart Price | Precificação de aluguel",
                   page_icon="🏠", layout="wide")

TIPOS_PT = {
    'APARTMENT': 'Apartamento', 'HOME': 'Casa', 'CONDOMINIUM': 'Casa de condomínio',
    'VILLAGE_HOUSE': 'Casa de vila', 'PENTHOUSE': 'Cobertura', 'FLAT': 'Flat',
    'KITNET': 'Kitnet/Studio', 'BUSINESS': 'Comercial', 'FARM': 'Sítio/Chácara',
    'ALLOTMENT_LAND': 'Terreno', 'RESIDENTIAL_ALLOTMENT_LAND': 'Terreno residencial',
    'TWO_STORY_HOUSE': 'Sobrado', 'RESIDENTIAL_BUILDING': 'Prédio residencial',
}
pt = lambda t: TIPOS_PT.get(t, str(t).replace('_', ' ').title())


def fmt_num(v, casas=1):
    """numero pt-BR: ponto de milhar, virgula decimal"""
    s = f"{v:,.{casas}f}"
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def fmt_moeda(v, casas=0):
    """R$ pt-BR pra usar em st.metric (nao renderiza markdown, cifrao normal)"""
    return f"R$ {fmt_num(v, casas)}"


def fmt_moeda_md(v, casas=0):
    """R$ pt-BR pra usar dentro de st.markdown/st.caption/st.info.
    Cifrao escapado com \\$ porque essas funcoes tratam $texto$ como LaTeX
    e comem o simbolo (era a causa do 'R716· R/m2' que sumia o R$)."""
    return f"R\\$ {fmt_num(v, casas)}"


# --------------------------------------------------------------- carregamento
@st.cache_resource(show_spinner=False)
def carregar_modelo():
    return joblib.load('model.joblib')


@st.cache_data(show_spinner=False)
def carregar_dados():
    dados = pd.read_parquet('dados_app.parquet')
    ref = pd.read_parquet('ref_bairro.parquet')
    dados['tipo_pt'] = dados.tipo.map(pt)
    return dados, ref


try:
    ART = carregar_modelo()
    DADOS, REF = carregar_dados()
except FileNotFoundError:
    st.error("Artefatos não encontrados. Rode `python treinar_modelo.py dataZAP.csv` "
             "para gerar `model.joblib`, `dados_app.parquet` e `ref_bairro.parquet`.")
    st.stop()

MET = ART['metricas']


# ------------------------------------------------------------------ previsao
def estimar_condo(cidade, bairro, tipo, area):
    """condominio por m2 em cascata: bairro+tipo -> cidade+tipo -> tipo"""
    a = ART
    v = (a['condo_m2_bairro_tipo'].get(f'{bairro}|{tipo}')
         or a['condo_m2_cidade_tipo'].get(f'{cidade}|{tipo}')
         or a['condo_m2_tipo'].get(tipo, 0.0))
    return float(v) * area


def estimar_iptu(cidade, bairro, tipo, area):
    a = ART
    v = (a['iptu_m2_bairro_tipo'].get(f'{bairro}|{tipo}')
         or a['iptu_m2_cidade_tipo'].get(f'{cidade}|{tipo}')
         or a['iptu_m2_tipo'].get(tipo, 0.0))
    return float(v) * area


def prever(cidade, bairro, tipo, area, quartos, banheiros, suites, vagas,
           condo=None, area_total=None, mobiliado=0, piscina=0, academia=0,
           churrasqueira=0, salao_festa=0, n_amen=0):
    """monta a linha exatamente no formato de treino e devolve o aluguel previsto"""
    a = ART
    condo_informado = condo is not None
    if not condo_informado:
        condo = estimar_condo(cidade, bairro, tipo, area)
    iptu = estimar_iptu(cidade, bairro, tipo, area)
    area_total = area_total or area
    lazer = piscina + academia + salao_festa

    linha = {f: 0 for f in a['feats']}
    linha.update({
        'area': area, 'area_total': area_total, 'quartos': quartos, 'banheiros': banheiros,
        'suites': suites, 'vagas': vagas, 'condo': condo, 'iptu': iptu,
        'custo_fixo': condo + iptu,
        'area_por_quarto': area / max(quartos, 1),
        'banh_por_quarto': banheiros / max(quartos, 1),
        'ratio_area': area_total / area,
        'tem_suite': int(suites > 0),
        'sem_condominio': int(condo == 0),
        'condo_imputado': int(not condo_informado),
        'iptu_imputado': 1,
        'n_amen': n_amen, 'lazer': lazer, 'mobiliado': mobiliado,
        'piscina': piscina, 'academia': academia,
        'churrasqueira': churrasqueira, 'salao_festa': salao_festa,
        'te_bairro': a['enc_bairro'].get(bairro, a['enc_bairro_geral']),
        'te_cidade': a['enc_cidade'].get(cidade, a['enc_cidade_geral']),
        'tipo_cod': a['cat_maps']['tipo'].get(tipo, -1),
        'zona_cod': -1,
    })
    ll = a['mediana_lat_lon'].get(f'{cidade}|{bairro}', {})
    linha['lat'] = ll.get('lat', np.nan)
    linha['lon'] = ll.get('lon', np.nan)

    x = pd.DataFrame([linha])[a['feats']]
    return float(np.expm1(a['modelo'].predict(x)[0])), condo, iptu


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🏠 Smart Price")
st.sidebar.caption("Precificação de aluguel residencial baseada em 21,6 mil anúncios reais")

CIDADE_TODAS = "🌎 Todas as cidades (Brasil)"
cidades = sorted(DADOS.cidade.unique())
opcoes_cidade = [CIDADE_TODAS] + cidades
idx_default = opcoes_cidade.index('São Paulo') if 'São Paulo' in opcoes_cidade else 0
cidade_sel = st.sidebar.selectbox("Cidade", opcoes_cidade, index=idx_default)
modo_brasil = cidade_sel == CIDADE_TODAS

d_cid = DADOS if modo_brasil else DADOS[DADOS.cidade == cidade_sel]

bairros_sel = []
if not modo_brasil:
    bairros_disp = sorted(d_cid.bairro.value_counts()[lambda s: s >= 5].index)
    bairros_sel = st.sidebar.multiselect("Bairro (vazio = todos)", bairros_disp)
else:
    st.sidebar.caption("Filtro de bairro fica disponível ao escolher uma cidade específica.")

tipos_disp = sorted(d_cid.tipo_pt.unique())
tipos_sel = st.sidebar.multiselect("Tipo de imóvel", tipos_disp, default=tipos_disp)

q_max = int(min(d_cid.quartos.max(), 6))
quartos_sel = st.sidebar.slider("Quartos", 0, q_max, (0, q_max))

a_lo, a_hi = int(d_cid.area.quantile(.01)), int(d_cid.area.quantile(.99))
area_sel = st.sidebar.slider("Área útil (m²)", a_lo, a_hi, (a_lo, a_hi))

p_hi = int(d_cid.preco.quantile(.99))
preco_sel = st.sidebar.slider("Faixa de aluguel (R$)", 0, p_hi, (0, p_hi), step=100)

f = d_cid[
    d_cid.tipo_pt.isin(tipos_sel)
    & d_cid.quartos.between(*quartos_sel)
    & d_cid.area.between(*area_sel)
    & d_cid.preco.between(*preco_sel)
]
if bairros_sel:
    f = f[f.bairro.isin(bairros_sel)]

st.sidebar.divider()
st.sidebar.metric("Anúncios no filtro", fmt_num(len(f), 0))
st.sidebar.caption(f"Modelo: LightGBM · MAPE {MET['MAPE_%']:.1f}% · R² {MET['R2']:.2f}")

# label amigavel usado nos titulos das abas
local_label = "Brasil (todas as cidades)" if modo_brasil else cidade_sel
# nivel de agregacao: no Brasil inteiro comparamos por cidade, numa cidade especifica por bairro
nivel_col = 'cidade' if modo_brasil else 'bairro'
nivel_nome = 'cidade' if modo_brasil else 'bairro'


# ---------------------------------------------------------------------- topo
st.title("Smart Price")
st.markdown("Estimativa de valor de aluguel a partir das características do imóvel e da localização.")

if len(f) == 0:
    st.warning("Nenhum anúncio no filtro atual. Amplie os critérios na barra lateral.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aluguel mediano", fmt_moeda(f.preco.median()))
c2.metric("R$ por m²", fmt_moeda((f.preco / f.area).median(), casas=1))
c3.metric("Área mediana", f"{fmt_num(f.area.median(), 0)} m²")
c4.metric("Cidades cobertas" if modo_brasil else "Bairros cobertos",
          fmt_num(f.cidade.nunique() if modo_brasil else f.bairro.nunique(), 0))

tab_sim, tab_pan, tab_mapa, tab_mod = st.tabs(
    ["🎯 Simulador de aluguel", "📊 Panorama do mercado", "🗺️ Mapa", "🤖 Sobre o modelo"])


# ------------------------------------------------------------- simulador
with tab_sim:
    st.subheader("Simule o valor de aluguel")
    st.caption("Preencha as características do imóvel. O condomínio é estimado pelo bairro "
               "quando você não souber informar.")

    e1, e2, e3 = st.columns(3)
    with e1:
        cid_in = st.selectbox("Cidade", cidades,
                              index=cidades.index(cidade_sel), key="sim_cidade")
        b_opts = sorted(DADOS[DADOS.cidade == cid_in].bairro.value_counts()[lambda s: s >= 5].index)
        bai_in = st.selectbox("Bairro", b_opts, key="sim_bairro")
        tipos_cod = sorted(DADOS[DADOS.cidade == cid_in].tipo.unique())
        tipo_in = st.selectbox("Tipo de imóvel", tipos_cod, format_func=pt, key="sim_tipo")
    with e2:
        area_in = st.number_input("Área útil (m²)", 10, 3000, 70, step=5)
        quartos_in = st.number_input("Quartos", 0, 10, 2)
        banh_in = st.number_input("Banheiros", 0, 10, 2)
    with e3:
        suites_in = st.number_input("Suítes", 0, 10, 1)
        vagas_in = st.number_input("Vagas de garagem", 0, 10, 1)
        mob_in = st.checkbox("Mobiliado")

    with st.expander("Condomínio e comodidades (opcional)"):
        g1, g2 = st.columns(2)
        with g1:
            sabe_condo = st.checkbox("Sei o valor do condomínio")
            condo_in = st.number_input("Condomínio mensal (R$)", 0, 20000, 500, step=50,
                                       disabled=not sabe_condo)
        with g2:
            piscina_in = st.checkbox("Piscina")
            acad_in = st.checkbox("Academia")
            churras_in = st.checkbox("Churrasqueira")
            salao_in = st.checkbox("Salão de festas")

    if st.button("Calcular aluguel sugerido", type="primary", width="stretch"):
        n_amen = sum([piscina_in, acad_in, churras_in, salao_in, mob_in])
        valor, condo_us, iptu_us = prever(
            cid_in, bai_in, tipo_in, area_in, quartos_in, banh_in, suites_in, vagas_in,
            condo=float(condo_in) if sabe_condo else None,
            mobiliado=int(mob_in), piscina=int(piscina_in), academia=int(acad_in),
            churrasqueira=int(churras_in), salao_festa=int(salao_in), n_amen=n_amen)

        mape = MET['MAPE_%'] / 100
        lo, hi = valor * (1 - mape), valor * (1 + mape)

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Aluguel sugerido", fmt_moeda(valor))
        r2.metric("Faixa de negociação", f"{fmt_moeda(lo)} – {fmt_moeda(hi)}")
        r3.metric("Custo total ao inquilino",
                  fmt_moeda(valor + condo_us),
                  help="Aluguel previsto + condomínio (informado ou estimado pelo bairro).")

        st.caption(f"Condomínio {'informado' if sabe_condo else 'estimado'}: "
                   f"{fmt_moeda_md(condo_us)} · R\\$/m² previsto: {fmt_moeda_md(valor / area_in, 1)}")

        # comparacao com o bairro
        ref = REF[(REF.cidade == cid_in) & (REF.bairro == bai_in)]
        comp = DADOS[(DADOS.cidade == cid_in) & (DADOS.bairro == bai_in)]
        if len(ref) and len(comp) >= 5:
            med = ref.aluguel_mediano.iloc[0]
            delta = (valor / med - 1) * 100
            st.info(
                f"**{bai_in}** tem aluguel mediano de {fmt_moeda_md(med)} e "
                f"{fmt_moeda_md(ref.rs_m2.iloc[0], 1)}/m² "
                f"({int(ref.anuncios.iloc[0])} anúncios). A sugestão está "
                f"{abs(delta):.0f}% {'acima' if delta > 0 else 'abaixo'} da mediana do bairro — "
                "a mediana mistura todos os tamanhos, então a diferença é esperada quando o imóvel "
                "simulado foge do padrão local.")

            fig = px.histogram(comp, x="preco", nbins=40,
                               title=f"Onde a sugestão cai dentro de {bai_in}",
                               labels={"preco": "Aluguel (R$)", "count": "Anúncios"})
            fig.add_vline(x=valor, line_dash="dash", line_color="#d9534f",
                          annotation_text="sugestão", annotation_position="top")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")

        mape_tipo = ART['mape_por_tipo'].get(tipo_in)
        aviso = (f"Erro médio do modelo para {pt(tipo_in).lower()}: {mape_tipo:.0f}%. "
                 if mape_tipo else "")
        if valor > 6000:
            aviso += ("Nesta faixa o modelo tende a subestimar (alto padrão tem amostra rala), "
                      "então trate a sugestão como piso de negociação. ")
        aviso += "Base de anúncios de 2020 — aplique reajuste por índice antes de fechar contrato."
        st.warning(aviso)


# --------------------------------------------------------------- panorama
with tab_pan:
    st.subheader(f"Mercado de aluguel — {local_label}")

    g1, g2 = st.columns(2)
    with g1:
        por_tipo = f.groupby('tipo_pt').agg(
            anuncios=('preco', 'size'), mediana=('preco', 'median')).reset_index()
        por_tipo = por_tipo[por_tipo.anuncios >= 5].sort_values('mediana')
        fig = px.bar(por_tipo, x='mediana', y='tipo_pt', orientation='h',
                     title="Aluguel mediano por tipo de imóvel",
                     labels={'mediana': 'R$', 'tipo_pt': ''}, text_auto='.0f')
        fig.update_traces(marker_color='#4682b4')
        st.plotly_chart(fig, width="stretch")

        fq = f[f.quartos <= 5]
        fig = px.box(fq, x='quartos', y='preco', points=False,
                     title="Distribuição do aluguel por número de quartos",
                     labels={'quartos': 'Quartos', 'preco': 'R$'})
        fig.update_traces(marker_color='#4682b4')
        fig.update_yaxes(range=[0, fq.preco.quantile(.97)])
        st.plotly_chart(fig, width="stretch")

    with g2:
        min_amostra = 30 if modo_brasil else 10
        b = f.groupby(nivel_col).agg(
            anuncios=('preco', 'size'), rs_m2=('rs_m2', 'median'),
            mediana=('preco', 'median')).reset_index()
        b = b[b.anuncios >= min_amostra].sort_values('rs_m2')
        if len(b) >= 4:
            extremos = pd.concat([b.head(7), b.tail(7)]).drop_duplicates(nivel_col)
            fig = px.bar(extremos, x='rs_m2', y=nivel_col, orientation='h',
                         title=f"{nivel_nome.capitalize()}s mais baratos e mais caros por m² "
                               f"(mín. {min_amostra} anúncios)",
                         labels={'rs_m2': 'R$/m²', nivel_col: ''}, text_auto='.1f',
                         color='rs_m2', color_continuous_scale='RdYlGn_r')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Amplitude dentro de {local_label}: "
                       f"**{fmt_num(b.rs_m2.max() / b.rs_m2.min())}x** entre o {nivel_nome} "
                       f"mais caro e o mais barato por m².")
        else:
            st.info(f"Poucos {nivel_nome}s com amostra suficiente no filtro atual.")

        fig = px.scatter(f.sample(min(3000, len(f)), random_state=1),
                         x='area', y='preco', color='tipo_pt', opacity=.5,
                         title="Área × aluguel",
                         labels={'area': 'Área útil (m²)', 'preco': 'R$', 'tipo_pt': 'Tipo'})
        fig.update_xaxes(range=[0, f.area.quantile(.97)])
        fig.update_yaxes(range=[0, f.preco.quantile(.97)])
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Ranking de bairros" + (" (todas as cidades)" if modo_brasil else ""))
    cols_tabela = ['cidade', 'bairro'] if modo_brasil else ['bairro']
    tabela = f.groupby(cols_tabela).agg(
        anúncios=('preco', 'size'), aluguel_mediano=('preco', 'median'),
        rs_m2=('rs_m2', 'median'), área_mediana=('area', 'median')).reset_index()
    tabela = tabela[tabela['anúncios'] >= 5].sort_values('aluguel_mediano', ascending=False)
    st.dataframe(tabela.round(1), width="stretch", hide_index=True)


# ------------------------------------------------------------------- mapa
with tab_mapa:
    titulo_mapa = "Preço médio por cidade — Brasil" if modo_brasil else f"Preço médio por bairro — {cidade_sel}"
    st.subheader(titulo_mapa)
    metrica = st.radio("Métrica exibida", ["R$ por m²", "Aluguel mediano"],
                       horizontal=True, label_visibility="collapsed")
    min_amostra_mapa = 20 if modo_brasil else 5

    mp = f.dropna(subset=['lat', 'lon']).groupby(nivel_col).agg(
        anuncios=('preco', 'size'), aluguel_mediano=('preco', 'median'),
        rs_m2=('rs_m2', 'median'), lat=('lat', 'median'), lon=('lon', 'median')).reset_index()
    mp = mp[mp.anuncios >= min_amostra_mapa]

    if len(mp) == 0:
        st.info("Sem coordenadas suficientes no filtro atual.")
    else:
        col = 'rs_m2' if metrica == "R$ por m²" else 'aluguel_mediano'
        fig = px.scatter_map(
            mp, lat='lat', lon='lon', color=col, size='anuncios',
            hover_name=nivel_col,
            hover_data={'anuncios': True, 'aluguel_mediano': ':.0f', 'rs_m2': ':.1f',
                        'lat': False, 'lon': False},
            color_continuous_scale='RdYlGn_r', size_max=35,
            zoom=3 if modo_brasil else 10, center={'lat': -14.2, 'lon': -51.9} if modo_brasil else None,
            map_style='carto-positron', height=600,
            labels={'rs_m2': 'R$/m²', 'aluguel_mediano': 'Aluguel mediano', 'anuncios': 'Anúncios'})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        legenda_unidade = "cidade" if modo_brasil else "bairro"
        st.caption(f"Cada bolha é uma {legenda_unidade}: o tamanho representa o volume de anúncios "
                   f"e a cor a métrica escolhida. {legenda_unidade.capitalize()}s com menos de "
                   f"{min_amostra_mapa} anúncios ficam fora.")
        if modo_brasil:
            st.caption("Escolha uma cidade específica na barra lateral para ver o detalhe por bairro.")


# ------------------------------------------------------------ sobre o modelo
with tab_mod:
    st.subheader("Como a estimativa é calculada")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", fmt_moeda(MET['MAE']), help="Erro absoluto médio em reais")
    m2.metric("MAPE", f"{fmt_num(MET['MAPE_%'], 1)}%", help="Erro percentual médio")
    m3.metric("RMSE", fmt_moeda(MET['RMSE']), help="Penaliza erros grandes")
    m4.metric("R²", fmt_num(MET['R2'], 3), help="Variação do preço explicada pelo modelo")

    p1, p2, p3 = st.columns(3)
    p1.metric("Previsões dentro de ±10%", f"{MET['dentro_10']:.0f}%")
    p2.metric("Dentro de ±20%", f"{MET['dentro_20']:.0f}%")
    p3.metric("Dentro de ±30%", f"{MET['dentro_30']:.0f}%")

    st.markdown(f"""
**Modelo.** LightGBM treinado no log do aluguel, escolhido em comparação com Regressão Linear,
Árvore de Decisão, Random Forest e XGBoost no mesmo split. O boosting empatou com o XGBoost dentro
do ruído e a escolha foi pelo RMSE menor (erra menos nos imóveis caros) e pelo peso menor no deploy.
A regressão linear foi descartada por extrapolar previsões de {fmt_moeda_md(1700000)} — inaceitável
num produto de precificação.

**Base.** 35,7 mil anúncios do ZAP Imóveis viraram 21,6 mil modeláveis: a sentinela `normal` virou
nulo, o separador de milhar foi corrigido antes de qualquer conta, 11,5 mil republicações do mesmo
anúncio saíram e ficaram as 56 cidades com amostra suficiente.

**O que mais pesa.** Latitude, longitude e a média do bairro ocupam o pódio de importância. Remover
localização piora o MAPE em quase 5 pontos; remover condomínio e IPTU custa quase nada — por isso o
simulador funciona bem mesmo sem você saber o condomínio.

**Limites conhecidos.**
- Acima de {fmt_moeda_md(6000)} o modelo subestima em média, então a sugestão é **piso de negociação**.
- Cobertura é o tipo menos previsível; flat é o mais previsível.
- A base é de julho de 2020 e precisa de reajuste por índice antes de virar número de contrato.
- Estado de conservação, andar, vista e idade do prédio não existem na base — dois apartamentos
  idênticos no mesmo prédio anunciam com 20% de diferença e nenhuma coluna aqui separa os dois.
""")

    st.markdown("##### Erro médio por tipo de imóvel")
    mt = pd.DataFrame([{'Tipo': pt(k), 'MAPE (%)': v} for k, v in ART['mape_por_tipo'].items()])
    st.dataframe(mt.sort_values('MAPE (%)'), width="stretch", hide_index=True)

st.divider()
st.caption("Smart Price · dados: ZAP Imóveis (Kaggle, jul/2020) · modelo LightGBM · "
           "ferramenta de apoio à decisão, não substitui avaliação profissional.")
