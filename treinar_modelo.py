"""
Smart Price - geracao dos artefatos consumidos pelo app.py

Reproduz o pipeline do notebook smart_price.ipynb (limpeza -> imputacao ->
feature engineering -> LightGBM) e grava:

    model.joblib        modelo final + todo o pre-processamento embutido
    dados_app.parquet   base modelavel enxuta, usada nos graficos e filtros
    ref_bairro.parquet  tabela de referencia por bairro (mapa e comparacoes)

Uso:  python treinar_modelo.py dataZAP.csv
"""
import sys, os, json, warnings
import numpy as np, pandas as pd, joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
CSV = sys.argv[1] if len(sys.argv) > 1 else 'dataZAP.csv'
SAIDA = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 1. carga
raw = pd.read_csv(CSV, sep=';', low_memory=False, dtype=str)
print(f'base bruta: {raw.shape[0]:,} x {raw.shape[1]}')

# 'normal' e a sentinela de nulo do dataset
df = raw.replace('normal', np.nan)

# o ponto e separador de MILHAR: '1.300' = mil e trezentos
def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace('.', '', regex=False), errors='coerce')

num_cols = ['listing.pricingInfo.rentalPrice', 'listing.pricingInfo.rentalTotalPrice',
            'listing.pricingInfo.monthlyCondoFee', 'listing.pricingInfo.yearlyIptu',
            'listing.usableAreas', 'listing.totalAreas', 'listing.bathrooms',
            'listing.bedrooms', 'listing.suites', 'listing.parkingSpaces']
for c in num_cols:
    df[c] = to_num(df[c])
df['lat'] = pd.to_numeric(df['listing.address.point.lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['listing.address.point.lon'], errors='coerce')

# ------------------------------------------------------------ 2. duplicatas
df = df.drop_duplicates(subset='listing.id')
chave = ['listing.address.city', 'listing.address.neighborhood', 'listing.pricingInfo.rentalPrice',
         'listing.usableAreas', 'listing.bedrooms', 'listing.bathrooms', 'account.name']
df = df.drop_duplicates(subset=chave)
df = df[df['listing.usageTypes'].str.contains('RESIDENTIAL', na=False)]
print(f'apos deduplicacao e filtro residencial: {len(df):,}')

# -------------------------------------------------------------- 3. outliers
p = df['listing.pricingInfo.rentalPrice']
a = df['listing.usableAreas']
r = p / a
mask = p.between(300, 50000) & a.between(10, 3000) & r.between(3, 200)
df = df[mask].copy()

df = df.rename(columns={
    'listing.pricingInfo.rentalPrice': 'preco', 'listing.usableAreas': 'area',
    'listing.totalAreas': 'area_total', 'listing.pricingInfo.monthlyCondoFee': 'condo',
    'listing.pricingInfo.yearlyIptu': 'iptu', 'listing.unitTypes': 'tipo',
    'listing.address.city': 'cidade', 'listing.address.neighborhood': 'bairro',
    'listing.address.state': 'uf', 'listing.address.zone': 'zona',
    'listing.bedrooms': 'quartos', 'listing.bathrooms': 'banheiros',
    'listing.suites': 'suites', 'listing.parkingSpaces': 'vagas',
    'listing.amenities': 'amenities', 'listing.furnished': 'mobiliado'})
print(f'base limpa: {len(df):,}')

# ------------------------------------------------------------- 4. imputacao
# casa de rua nao paga condominio: nulo ali significa zero
casa = df.tipo.isin(['HOME', 'VILLAGE_HOUSE'])
df['sem_condominio'] = (casa & df.condo.isna()).astype(int)
df.loc[casa & df.condo.isna(), 'condo'] = 0

def cascata(df_base, valor, chaves_list, min_n=10):
    """mediana por grupo com minimo de amostra, caindo pra chave mais grossa"""
    out = pd.Series(np.nan, index=df_base.index)
    tabelas = {}
    for chaves in chaves_list:
        g = valor.groupby([df_base[k] for k in chaves]).agg(['median', 'size'])
        g = g[g['size'] >= min_n]['median']
        tabelas[tuple(chaves)] = g
        idx = (pd.MultiIndex.from_arrays([df_base[k] for k in chaves])
               if len(chaves) > 1 else pd.Index(df_base[chaves[0]]))
        out = out.fillna(pd.Series(g.reindex(idx).values, index=df_base.index))
    return out, tabelas

cpm_full, tab_condo = cascata(df, (df.condo / df.area), [['bairro', 'tipo'], ['cidade', 'tipo'], ['tipo']])
df['condo_imputado'] = df.condo.isna().astype(int)
df['condo'] = df.condo.fillna(cpm_full * df.area)

ipm_full, tab_iptu = cascata(df, (df.iptu / df.area), [['bairro', 'tipo'], ['cidade', 'tipo'], ['tipo']])
df['iptu_imputado'] = df.iptu.isna().astype(int)
df['iptu'] = df.iptu.fillna(ipm_full * df.area)

# quem tem suite/vaga anuncia; nulo = nao tem
df['suites'] = df.suites.fillna(0)
df['vagas'] = df.vagas.fillna(0)
df['area_total'] = df.area_total.fillna(df.area)
df['bairro'] = df.bairro.fillna('NAO_INFORMADO')
df['zona'] = df.zona.fillna('NAO_INFORMADO')
df['amenities'] = df.amenities.fillna('')
df['condo'] = df.condo.fillna(0)
df['iptu'] = df.iptu.fillna(0)

# --------------------------------------------------------- 5. comodidades
amen_flags = {'listing.pool': 'piscina', 'listing.gym': 'academia', 'listing.barbgrill': 'churrasqueira',
              'listing.partyhall': 'salao_festa', 'listing.playground': 'playground',
              'listing.sportcourt': 'quadra', 'listing.sauna': 'sauna', 'listing.backyard': 'quintal',
              'listing.garden': 'jardim', 'listing.guestpark': 'vaga_visitante',
              'listing.fireplace': 'lareira', 'listing.bathtub': 'banheira', 'listing.hottub': 'hidro',
              'listing.mountainview': 'vista_montanha', 'listing.tenniscourt': 'quadra_tenis',
              'listing.soundproofing': 'isolamento', 'mobiliado': 'mobiliado'}
for c, n in amen_flags.items():
    df[n] = df[c].astype(str).str.lower().eq('true').astype(int)
df['n_amen'] = df.amenities.str.count(r'\|') + (df.amenities.str.len() > 0).astype(int)

# ------------------------------------------------- 6. recorte geografico
CORTE = 50
vc = df.cidade.value_counts()
base = df[df.cidade.isin(vc[vc >= CORTE].index)].copy()
print(f'base modelavel: {len(base):,} | {base.cidade.nunique()} cidades | {base.bairro.nunique():,} bairros')

# ------------------------------------------------ 7. feature engineering
base['lazer'] = base[['piscina', 'academia', 'salao_festa', 'playground', 'quadra', 'sauna']].sum(axis=1)
base['area_por_quarto'] = base.area / base.quartos.replace(0, 1)
base['banh_por_quarto'] = base.banheiros / base.quartos.replace(0, 1)
base['ratio_area'] = base.area_total / base.area
base['tem_suite'] = (base.suites > 0).astype(int)
base['custo_fixo'] = base.condo + base.iptu
base['y'] = np.log1p(base.preco)

FEATS = ['area', 'area_total', 'quartos', 'banheiros', 'suites', 'vagas', 'condo', 'iptu', 'custo_fixo',
         'area_por_quarto', 'banh_por_quarto', 'ratio_area', 'tem_suite', 'lat', 'lon',
         'n_amen', 'lazer', 'mobiliado', 'piscina', 'academia', 'churrasqueira', 'salao_festa',
         'sem_condominio', 'condo_imputado', 'iptu_imputado', 'te_bairro', 'te_cidade',
         'tipo_cod', 'zona_cod']

def encode_alvo(treino, alvo, chave, suavizacao=20):
    """media do log-preco por chave, puxada pra media geral quando o grupo tem pouca amostra"""
    geral = alvo.mean()
    g = alvo.groupby(treino[chave]).agg(['mean', 'size'])
    return ((g['mean'] * g['size'] + geral * suavizacao) / (g['size'] + suavizacao)), geral

tr, te = train_test_split(base, test_size=.2, random_state=42)

cat_maps = {}
for c in ['tipo', 'zona']:
    cats = sorted(tr[c].dropna().unique())
    cat_maps[c] = {v: i for i, v in enumerate(cats)}
    tr[c + '_cod'] = tr[c].map(cat_maps[c]).fillna(-1).astype(int)
    te[c + '_cod'] = te[c].map(cat_maps[c]).fillna(-1).astype(int)

enc_maps = {}
for chave in ['bairro', 'cidade']:
    oof = pd.Series(index=tr.index, dtype=float)
    for i_in, i_out in KFold(5, shuffle=True, random_state=1).split(tr):
        dentro, fora = tr.iloc[i_in], tr.iloc[i_out]
        mapa, geral = encode_alvo(dentro, dentro.y, chave)
        oof.iloc[i_out] = fora[chave].map(mapa).fillna(geral).values
    mapa_full, geral = encode_alvo(tr, tr.y, chave)
    enc_maps[chave] = (mapa_full, geral)
    tr['te_' + chave] = oof
    te['te_' + chave] = te[chave].map(mapa_full).fillna(geral).values

# ---------------------------------------------------------- 8. modelagem
PARAMS = dict(n_estimators=900, learning_rate=.05, num_leaves=63, min_child_samples=20,
              subsample=.8, colsample_bytree=.8, random_state=42, verbosity=-1)
mod = lgb.LGBMRegressor(**PARAMS).fit(tr[FEATS], tr.y)
real = np.expm1(te.y)
pred = np.expm1(mod.predict(te[FEATS]))
metricas = {'modelo': 'LightGBM',
            'RMSE': float(np.sqrt(mean_squared_error(real, pred))),
            'MAE': float(mean_absolute_error(real, pred)),
            'MAPE_%': float(np.mean(np.abs(real - pred) / real) * 100),
            'R2': float(r2_score(real, pred))}
erro_pct = (pred - real).abs() / real * 100
metricas['dentro_10'] = float((erro_pct <= 10).mean() * 100)
metricas['dentro_20'] = float((erro_pct <= 20).mean() * 100)
metricas['dentro_30'] = float((erro_pct <= 30).mean() * 100)
print(json.dumps(metricas, indent=2))

por_tipo = te.assign(e=erro_pct).groupby('tipo').e.mean().round(1).to_dict()

# ------------------------------------- 9. modelo final na base completa
X_full, y_full = base.copy(), base.y
for c in ['tipo', 'zona']:
    X_full[c + '_cod'] = X_full[c].map(cat_maps[c]).fillna(-1).astype(int)
for chave in ['bairro', 'cidade']:
    mapa, geral = encode_alvo(X_full, y_full, chave)
    enc_maps[chave] = (mapa, geral)
    X_full['te_' + chave] = X_full[chave].map(mapa).fillna(geral)

modelo_final = lgb.LGBMRegressor(**PARAMS).fit(X_full[FEATS], y_full)
print(f'modelo final treinado em {len(X_full):,} anuncios')

# ------------------------------------------------------- 10. exportacao
ref_bairro = base.groupby(['cidade', 'bairro']).apply(lambda g: pd.Series({
    'anuncios': len(g), 'aluguel_mediano': g.preco.median(), 'rs_m2': (g.preco / g.area).median(),
    'condo_m2': (g.condo / g.area).median(), 'lat': g.lat.median(), 'lon': g.lon.median()
})).reset_index()

def chaves_str(d):
    """joblib com chave tupla funciona, mas normalizo pra str|str por seguranca"""
    return {('|'.join(map(str, k)) if isinstance(k, tuple) else str(k)): float(v)
            for k, v in d.items() if pd.notna(v)}

artefatos = {
    'modelo': modelo_final,
    'feats': FEATS,
    'cat_maps': cat_maps,
    'enc_bairro': {str(k): float(v) for k, v in enc_maps['bairro'][0].to_dict().items()},
    'enc_bairro_geral': float(enc_maps['bairro'][1]),
    'enc_cidade': {str(k): float(v) for k, v in enc_maps['cidade'][0].to_dict().items()},
    'enc_cidade_geral': float(enc_maps['cidade'][1]),
    'condo_m2_bairro_tipo': chaves_str(tab_condo[('bairro', 'tipo')].to_dict()),
    'condo_m2_cidade_tipo': chaves_str(tab_condo[('cidade', 'tipo')].to_dict()),
    'condo_m2_tipo': chaves_str(tab_condo[('tipo',)].to_dict()),
    'iptu_m2_bairro_tipo': chaves_str(tab_iptu[('bairro', 'tipo')].to_dict()),
    'iptu_m2_cidade_tipo': chaves_str(tab_iptu[('cidade', 'tipo')].to_dict()),
    'iptu_m2_tipo': chaves_str(tab_iptu[('tipo',)].to_dict()),
    'mediana_lat_lon': {f'{c}|{b}': {'lat': v['lat'], 'lon': v['lon']}
                        for (c, b), v in base.groupby(['cidade', 'bairro'])[['lat', 'lon']]
                        .median().to_dict('index').items()},
    'metricas': metricas,
    'mape_por_tipo': por_tipo,
}
joblib.dump(artefatos, os.path.join(SAIDA, 'model.joblib'), compress=3)

cols_app = ['cidade', 'bairro', 'zona', 'uf', 'tipo', 'preco', 'area', 'area_total', 'quartos',
            'banheiros', 'suites', 'vagas', 'condo', 'iptu', 'n_amen', 'lazer', 'mobiliado',
            'piscina', 'academia', 'lat', 'lon']
dados_app = base[cols_app].copy()
dados_app['rs_m2'] = (dados_app.preco / dados_app.area).round(1)
dados_app.to_parquet(os.path.join(SAIDA, 'dados_app.parquet'), index=False, compression='zstd')
ref_bairro.to_parquet(os.path.join(SAIDA, 'ref_bairro.parquet'), index=False, compression='zstd')

for f in ['model.joblib', 'dados_app.parquet', 'ref_bairro.parquet']:
    print(f'{f:22s} {os.path.getsize(os.path.join(SAIDA, f)) / 1024**2:.1f} MB')
