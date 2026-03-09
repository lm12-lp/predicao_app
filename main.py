import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import json
import os

USERS_FILE = "users.json"

# ----------- Funções de suporte -----------

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def create_account(username, password):
    users = load_users()
    if username in users:
        return False  # Já existe
    users[username] = password
    save_users(users)
    return True

def validate_login(username, password):
    users = load_users()
    return users.get(username) == password

# ----------- Tela de Login e Criação -----------

def login_screen():
    st.title("🔐 Login - Boost Sell Out")

    mode = st.radio("Selecione uma opção:", ["Entrar", "Criar conta"])

    if mode == "Entrar":
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if validate_login(username, password):
                st.session_state["auth"] = True
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos.")
    else:  # Criar conta
        new_user = st.text_input("Novo usuário")
        new_pass = st.text_input("Nova senha", type="password")
        if st.button("Criar conta"):
            if create_account(new_user, new_pass):
                st.success("Conta criada! Agora faça login.")
            else:
                st.error("Este usuário já existe.")

def require_login():
    if "auth" not in st.session_state or st.session_state["auth"] is not True:
        login_screen()
        st.stop()

# 🔒 EXIGIR LOGIN ANTES DE MOSTRAR O DASHBOARD
require_login()

# =========================
# === A PARTIR DAQUI É O SEU APP ORIGINAL ===
# =========================

def get_data_limite(reference_date=None):
    """Aplica a regra D-4 (sex a dom) / D-6 (seg a qui) a partir da data de referência."""
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    dia_semana = reference_date.dayofweek  # 0=seg ... 6=dom
    dias_atras = 4 if dia_semana >= 4 else 6
    data_limite = reference_date - pd.Timedelta(days=dias_atras)
    return data_limite, dias_atras

def read_boost_sheet(file_uploaded, sheet_name):
    """Lê cada aba do template Boost e retorna df com colunas ['date','y']."""
    try:
        df = pd.read_excel(file_uploaded, sheet_name=sheet_name, skiprows=9, usecols='B:D')
        df.columns = ['Ano_Mes', 'Dia_Mes', 'Total']

        # Limpeza de linhas não-data
        df = df.dropna(subset=['Ano_Mes', 'Dia_Mes'], how='all')
        df = df[~df['Ano_Mes'].astype(str).str.contains('Ano|Mês|Month', case=False, na=False)]

        # Dia do mês como número
        df['Dia_Mes'] = pd.to_numeric(df['Dia_Mes'], errors='coerce')
        df = df.dropna(subset=['Dia_Mes'])

        # Construir data (duas tentativas)
        df['Ano_Mes_str'] = df['Ano_Mes'].astype(str).str.strip()
        df['date'] = None
        try:
            # Ex.: Dia-Mês-Ano (com Ano_Mes em "mm-YYYY" ou "m-YYYY")
            df['date'] = pd.to_datetime(
                df['Dia_Mes'].astype(int).astype(str) + '-' + df['Ano_Mes_str'],
                format='%d-%m-%Y',
                errors='coerce'
            )
        except Exception:
            pass

        # Se falhou: converter Ano_Mes para datetime e somar Dia-1
        if df['date'].isna().all():
            try:
                ano_mes_date = pd.to_datetime(df['Ano_Mes'], errors='coerce')
                df['date'] = ano_mes_date + pd.to_timedelta(df['Dia_Mes'].astype(int) - 1, unit='D')
            except Exception:
                pass

        df = df.dropna(subset=['date'])

        # Total como número
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        df = df.dropna(subset=['Total'])

        df = df[['date', 'Total']].copy()
        df.columns = ['date', 'y']
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Erro ao ler aba {sheet_name}: {str(e)}")
        return None

def build_exog(df, date_col='date'):
    """Cria variáveis exógenas de calendário."""
    out = df.copy()
    out['year'] = out[date_col].dt.year
    out['month'] = out[date_col].dt.month
    out['day'] = out[date_col].dt.day
    out['dow'] = out[date_col].dt.dayofweek
    month_end_day = out[date_col].dt.days_in_month
    out['days_to_month_end'] = month_end_day - out['day']
    out['is_last5'] = (out['days_to_month_end'] <= 4).astype(int)
    out['is_weekend'] = out['dow'].isin([5, 6]).astype(int)
    out['is_bday'] = (out['dow'] < 5).astype(int)
    out['month_progress'] = out['day'] / month_end_day
    return out

def supervised_from_series(y, exog_df, lags=14):
    """Transforma série em supervised (lags + exógenas)."""
    y = pd.Series(y).reset_index(drop=True)
    X_lags = {}
    for i in range(1, lags + 1):
        X_lags[f'lag_{i}'] = y.shift(i)
    X_lags = pd.DataFrame(X_lags)
    X = pd.concat([X_lags, exog_df.reset_index(drop=True)], axis=1)
    X['y'] = y.values
    X = X.dropna().reset_index(drop=True)
    y_out = X.pop('y').values
    return X, y_out

def fit_predict_mlp(y_series, exog_hist, exog_future, horizon, lags=14):
    """Treina MLP com lags + exógenas e faz previsão recursiva dia a dia."""
    X_hist, y_hist = supervised_from_series(y_series, exog_hist.iloc[:len(y_series)], lags=lags)
    num_cols = X_hist.columns.tolist()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                             random_state=SEED, max_iter=2000))
    ])
    pipe.fit(X_hist[num_cols], y_hist)

    history = list(y_series.values)
    preds = []
    for h in range(horizon):
        ex_h = exog_future.iloc[h:h+1].reset_index(drop=True)

        last_vals = history[-lags:][::-1]
        if len(last_vals) < lags:
            last_vals = ([history[-1]] * (lags - len(last_vals))) + last_vals

        row = {f'lag_{i+1}': last_vals[i] for i in range(lags)}
        row_df = pd.DataFrame([row])
        row_df = pd.concat([row_df, ex_h], axis=1)
        row_df = row_df.reindex(columns=num_cols, fill_value=0)

        y_hat = float(pipe.predict(row_df)[0])
        # Alimenta a recursão com o valor já "clipado" (sem negativos)
        y_hat_clipped = max(y_hat, 0.0)
        preds.append(y_hat_clipped)
        history.append(y_hat_clipped)

    return np.array(preds)

def cumulative(arr):
    return np.cumsum(np.asarray(arr))

def calcular_metricas_mtd(df_real, df_pred, data_limite, target_month_year):
    """
    Calcula MTD esperado/realizado (até data_limite), MTG necessário (FLAT = total previsto),
    diferença e evolução (%).
    """
    year, month = target_month_year
    mask1 = df_real['date'].dt.year == year
    mask2 = df_real['date'].dt.month == month
    mask3 = df_real['date'] <= data_limite

    mtd_realizado = df_real[mask1 & mask2 & mask3]['y'].sum()

    dia_atual = min(data_limite.day, len(df_pred))
    mtd_esperado = df_pred.iloc[:dia_atual]['y_pred'].sum()

    mtg_necessario = df_pred['y_pred'].sum()  # FLAT previsto (mês cheio)
    diff = mtd_realizado - mtd_esperado
    evolucao = (diff / mtd_esperado * 100) if mtd_esperado > 0 else 0

    return {
        'mtd_esperado': float(mtd_esperado),
        'mtd_realizado': float(mtd_realizado),
        'mtg_necessario': float(mtg_necessario),
        'diff': float(diff),
        'evolucao': float(evolucao)
    }

def processar_canal(df, canal_name, data_limite, sel_year, sel_month, lags, last5_mult):
    """Treina, prevê e devolve (forecast_df, real_month, metricas) para o canal."""
    if df is None or df.empty:
        return None

    df_ate_limite = df[df['date'] <= data_limite].copy()
    if df_ate_limite.empty:
        st.warning(f"Sem dados para {canal_name}")
        return None

    horizon = calendar.monthrange(sel_year, sel_month)[1]

    hist_exog = build_exog(df_ate_limite, 'date')

    fut_dates = pd.date_range(datetime(sel_year, sel_month, 1), periods=horizon, freq='D')
    fut_df = pd.DataFrame({'date': fut_dates})
    fut_exog_full = build_exog(fut_df, 'date')

    exog_cols = ['year','month','day','dow','is_last5','is_weekend','is_bday','month_progress']
    hist_exog['is_last5_x'] = hist_exog['is_last5'] * last5_mult
    fut_exog_full['is_last5_x'] = fut_exog_full['is_last5'] * last5_mult
    exog_cols = exog_cols + ['is_last5_x']

    y_series = df_ate_limite['y'].astype(float)

    preds = fit_predict_mlp(
        y_series=y_series,
        exog_hist=hist_exog[exog_cols],
        exog_future=fut_exog_full[exog_cols],
        horizon=horizon,
        lags=lags
    )

    forecast_df = pd.DataFrame({'date': fut_dates, 'y_pred': preds})

    mask1 = df['date'].dt.year == sel_year
    mask2 = df['date'].dt.month == sel_month
    mask3 = df['date'] <= data_limite
    real_month = df[mask1 & mask2 & mask3].copy().sort_values('date')

    metricas = calcular_metricas_mtd(df, forecast_df, data_limite, (sel_year, sel_month))

    return {
        'forecast_df': forecast_df,
        'real_month': real_month,
        'metricas': metricas
    }

# ====== UI PRINCIPAL (apenas se autenticado) ======
st.title("📊 Boost Sell-Out Dashboard")
st.caption("Análise dos canais OFICIAL, INDIRETO e DIRETO")

# --- Configuração global de semente ---
SEED = 42
np.random.seed(SEED)  # para reprodutibilidade em numpy

with st.sidebar:
    st.header("📁 Upload")
    up = st.file_uploader("Template Boost (.xlsx)", type=['xlsx', 'xls'])
    st.divider()
    st.header("⚙ Configurações")
    data_ref = st.date_input("Data de referência", value=pd.Timestamp.now())
    data_limite, dias_regra = get_data_limite(pd.Timestamp(data_ref))
    st.info(f"Regra D-{dias_regra}")
    st.caption(f"Dados até: {data_limite.strftime('%d/%m/%Y')}")
    st.divider()
    lags = st.slider("Lags", 7, 35, 14)
    last5_multiplier = st.slider("Força últimos 5 dias", 1.0, 2.0, 1.2, 0.05)

if up is None:
    st.info("👈 Carregue o arquivo Excel")
    st.stop()

# -------- Leitura das abas --------
with st.spinner("Lendo arquivo..."):
    df_oficial  = read_boost_sheet(up, 'DPGP (OFICIAL)')
    df_indireto = read_boost_sheet(up, 'DPGP (INDIRETO)')
    df_direto   = read_boost_sheet(up, 'DPGP (DIRETO)')

abas_disponiveis = []
if df_oficial is not None:  abas_disponiveis.append(("OFICIAL",  df_oficial))
if df_indireto is not None: abas_disponiveis.append(("INDIRETO", df_indireto))
if df_direto is not None:   abas_disponiveis.append(("DIRETO",   df_direto))

if not abas_disponiveis:
    st.error("Nenhuma aba lida")
    st.stop()

st.success(f"{len(abas_disponiveis)} canal(is) carregado(s)")

# ---------- Seleção do mês-alvo ----------
with st.sidebar:
    st.divider()
    st.header("🎯 Mês")
    canal_temp, df_temp = abas_disponiveis[0]
    min_dt = df_temp['date'].min()
    max_dt = df_temp['date'].max() + pd.offsets.MonthBegin(1)
    months = pd.period_range(min_dt.to_period('M'), (max_dt + pd.offsets.MonthEnd(3)).to_period('M'), freq='M')
    month_strs = [f"{p.year}-{p.month:02d}" for p in months]
    sel = st.selectbox("Mês-alvo", options=month_strs, index=len(month_strs)-1)
    sel_year, sel_month = map(int, sel.split('-'))

# ---------- Pré-processar/Cache local dos resultados por canal ----------
dfs_by_name = {nome: df for (nome, df) in abas_disponiveis}
resultados = {}
for canal_name, df_canal in abas_disponiveis:
    resultados[canal_name] = processar_canal(
        df_canal, canal_name, data_limite, sel_year, sel_month, lags, last5_multiplier
    )

# ---------- Tabs: canais + Resultados Tabulados ----------
tab_names  = [nome for nome, _ in abas_disponiveis]
tab_labels = [f"📈 {nome}" for nome in tab_names] + ["📑 Resultados Tabulados"]
tabs = st.tabs(tab_labels)

# ---- Abas dos canais individuais ----
for idx, canal_name in enumerate(tab_names):
    df_canal = dfs_by_name[canal_name]
    resultado = resultados.get(canal_name)

    with tabs[idx]:
        st.header(f"Canal: {canal_name}")
        if not resultado:
            st.warning("Sem resultados para este canal.")
            continue

        forecast_df = resultado['forecast_df']
        real_month = resultado['real_month']
        metricas = resultado['metricas']

        st.subheader("📊 Métricas")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("MTD Esperado",  f"{metricas['mtd_esperado']:,.0f}")
        with col2: st.metric("MTD Realizado", f"{metricas['mtd_realizado']:,.0f}", delta=f"{metricas['evolucao']:+.1f}%")
        with col3: st.metric("Diferença",     f"{metricas['diff']:+,.0f}")
        with col4: st.metric("MTG Necessário (FLAT)", f"{metricas['mtg_necessario']:,.0f}")

        st.subheader("📈 Acumulado")
        fig1 = go.Figure()
        pred_cum = cumulative(forecast_df['y_pred'].values)
        fig1.add_trace(go.Scatter(x=forecast_df['date'], y=pred_cum, mode='lines+markers',
                                  name='Previsto', line=dict(color='#4A74F3', width=3)))
        if not real_month.empty:
            real_cum = cumulative(real_month['y'].values)
            fig1.add_trace(go.Scatter(x=real_month['date'], y=real_cum, mode='lines+markers',
                                      name='Realizado', line=dict(color='#9D5CE6', width=3)))
        fig1.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📊 Diário")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=forecast_df['date'], y=forecast_df['y_pred'],
                              name='Previsto', marker_color='#6C8BE0'))
        if not real_month.empty:
            fig2.add_trace(go.Scatter(x=real_month['date'], y=real_month['y'], name='Realizado',
                                      mode='lines+markers', line=dict(color='#B55CE6', width=3)))
        fig2.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Dados"):
            output_df = forecast_df.copy()
            output_df['pred_cum'] = pred_cum
            if not real_month.empty:
                output_df = output_df.merge(
                    real_month[['date', 'y']].rename(columns={'y': 'real'}),
                    on='date', how='left'
                )
                output_df['real_cum'] = output_df['real'].cumsum()
            output_df['date'] = output_df['date'].dt.strftime('%d/%m/%Y')
            st.dataframe(output_df, use_container_width=True)

        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Baixar {canal_name}", data=csv_data, file_name=f"previsao_{canal_name}.csv")

# ---- Comparativo (fora das tabs) ----
if len(tab_names) > 1:
    st.divider()
    st.header("🔄 Comparativo")
    fig_comp = go.Figure()
    for canal_name in tab_names:
        res = resultados.get(canal_name)
        if res:
            pred_cum = cumulative(res['forecast_df']['y_pred'].values)
            fig_comp.add_trace(go.Scatter(
                x=res['forecast_df']['date'], y=pred_cum,
                mode='lines+markers', name=canal_name
            ))
    fig_comp.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================
#   📑 Resultados Tabulados (Resumo Executivo)
# ============================

def _compute_baselines_2025(df_canal, sel_year, sel_month, data_limite):
    """Retorna (MTD_2025_alinhado, MTG_2025_cheio) para o mês-alvo."""
    if df_canal is None or df_canal.empty:
        return 0.0, 0.0
    year_25 = sel_year - 1
    last_day_25 = calendar.monthrange(year_25, sel_month)[1]
    corte = min(data_limite.day, last_day_25)

    mask_mes_25 = (df_canal['date'].dt.year == year_25) & (df_canal['date'].dt.month == sel_month)
    mtd_25 = float(df_canal[mask_mes_25 & (df_canal['date'].dt.day <= corte)]['y'].sum())
    mtg_25 = float(df_canal[mask_mes_25]['y'].sum())
    return mtd_25, mtg_25

def _row_linha1(rotulo, sellout_tt, mtd_esp_26, mtd_real_26, mtd_real_25):
    """Linha 1 — MTD (Realizado x Esperado)."""
    diff = mtd_real_26 - mtd_esp_26
    evol_real = ((mtd_real_26 - mtd_real_25) / mtd_real_25 * 100) if mtd_real_25 > 0 else 0.0
    return {
        "Cenário": rotulo,
        "Sell Out Un. TT": sellout_tt,                 # por padrão = FLAT (total previsto mês)
        "MTD esperado 2026": mtd_esp_26,
        "MTD realizado 2026": mtd_real_26,
        "Diff": diff,
        "MTD realizado 2025": mtd_real_25,
        "Evolução realizada": evol_real,
        "MTG necessário FLAT": None,                  # vazio na linha 1
        "MTG realizado em 2025": None,
        "Evolução necessária PARA FLAT": None,
        "Evolução prevista": None
    }

def _row_linha2(rotulo, sellout_tt, mtg_flat_26, mtg_real_25):
    """Linha 2 — FLAT (Previsto mês cheio x Real 2025 mês cheio)."""
    evol_flat = ((mtg_flat_26 - mtg_real_25) / mtg_real_25 * 100) if mtg_real_25 > 0 else 0.0
    evol_prevista = evol_flat  # conforme solicitado: baseada no final previsto do mês
    return {
        "Cenário": "",                                 # 'mesclado' visual
        "Sell Out Un. TT": sellout_tt,                 # por padrão = FLAT
        "MTD esperado 2026": None,
        "MTD realizado 2026": None,
        "Diff": None,
        "MTD realizado 2025": None,
        "Evolução realizada": None,
        "MTG necessário FLAT": mtg_flat_26,
        "MTG realizado em 2025": mtg_real_25,
        "Evolução necessária PARA FLAT": evol_flat,
        "Evolução prevista": evol_prevista
    }

def _fmt_number(x):
    if x is None or x == "": return ""
    try:
        return f"{x:,.0f}"
    except Exception:
        return x

def _fmt_percent(x):
    if x is None or x == "": return ""
    try:
        return f"{x:,.1f}%"
    except Exception:
        return x

with tabs[-1]:
    data_ref_str = pd.to_datetime(data_ref).strftime("%d/%m/%Y")
    data_lim_str = pd.to_datetime(data_limite).strftime("%d/%m/%Y")
    st.subheader(f"Cenário {data_ref_str} - prévia até {data_lim_str}")

    linhas = []

    # ----- DPGP = DIRETO + INDIRETO (pela predição) -----
    if "DIRETO" in resultados and "INDIRETO" in resultados:
        res_dir = resultados["DIRETO"]
        res_ind = resultados["INDIRETO"]
        df_dir = dfs_by_name["DIRETO"]
        df_ind = dfs_by_name["INDIRETO"]

        pred_dir = res_dir["forecast_df"]["y_pred"]
        pred_ind = res_ind["forecast_df"]["y_pred"]
        pred_dpgp = pred_dir + pred_ind

        dia_corte = min(data_limite.day, len(pred_dpgp))
        mtd_esp_26_dpgp = float(pred_dpgp.iloc[:dia_corte].sum())
        mtd_real_26_dpgp = float(res_dir["metricas"]["mtd_realizado"] + res_ind["metricas"]["mtd_realizado"])
        mtg_flat_26_dpgp = float(pred_dpgp.sum())

        mtd_25_dir, mtg_25_dir = _compute_baselines_2025(df_dir, sel_year, sel_month, data_limite)
        mtd_25_ind, mtg_25_ind = _compute_baselines_2025(df_ind, sel_year, sel_month, data_limite)
        mtd_25_dpgp = mtd_25_dir + mtd_25_ind
        mtg_25_dpgp = mtg_25_dir + mtg_25_ind

        # Sell Out Un. TT (por padrão, FLAT previsto do mês)
        sellout_tt_dpgp = mtg_flat_26_dpgp

        linhas.append(_row_linha1("Cenário -2,0% para DPGP", sellout_tt_dpgp, mtd_esp_26_dpgp, mtd_real_26_dpgp, mtd_25_dpgp))
        linhas.append(_row_linha2("Cenário -2,0% para DPGP", sellout_tt_dpgp, mtg_flat_26_dpgp, mtg_25_dpgp))
    else:
        st.warning("Para calcular DPGP, carregue os canais DIRETO e INDIRETO.")

    # ----- DIRETO -----
    if "DIRETO" in resultados:
        res = resultados["DIRETO"]
        df_c = dfs_by_name["DIRETO"]
        pred = res["forecast_df"]["y_pred"]
        dia_corte = min(data_limite.day, len(pred))
        mtd_esp_26 = float(pred.iloc[:dia_corte].sum())
        mtd_real_26 = float(res["metricas"]["mtd_realizado"])
        mtg_flat_26 = float(pred.sum())
        mtd_25, mtg_25 = _compute_baselines_2025(df_c, sel_year, sel_month, data_limite)
        sellout_tt = mtg_flat_26  # ajuste se quiser outro conceito de TT

        linhas.append(_row_linha1("Cenário -8,8% para canal DIRETO", sellout_tt, mtd_esp_26, mtd_real_26, mtd_25))
        linhas.append(_row_linha2("Cenário -8,8% para canal DIRETO", sellout_tt, mtg_flat_26, mtg_25))

    # ----- INDIRETO -----
    if "INDIRETO" in resultados:
        res = resultados["INDIRETO"]
        df_c = dfs_by_name["INDIRETO"]
        pred = res["forecast_df"]["y_pred"]
        dia_corte = min(data_limite.day, len(pred))
        mtd_esp_26 = float(pred.iloc[:dia_corte].sum())
        mtd_real_26 = float(res["metricas"]["mtd_realizado"])
        mtg_flat_26 = float(pred.sum())
        mtd_25, mtg_25 = _compute_baselines_2025(df_c, sel_year, sel_month, data_limite)
        sellout_tt = mtg_flat_26

        linhas.append(_row_linha1("Previsão de -0,2% para canal INDIRETO", sellout_tt, mtd_esp_26, mtd_real_26, mtd_25))
        linhas.append(_row_linha2("Previsão de -0,2% para canal INDIRETO", sellout_tt, mtg_flat_26, mtg_25))

    # ----- Montar DataFrame final -----
    df_exec = pd.DataFrame(linhas, columns=[
        "Cenário",
        "Sell Out Un. TT",
        "MTD esperado 2026",
        "MTD realizado 2026",
        "Diff",
        "MTD realizado 2025",
        "Evolução realizada",
        "MTG necessário FLAT",
        "MTG realizado em 2025",
        "Evolução necessária PARA FLAT",
        "Evolução prevista"
    ])

    # Formatação (números e %)
    num_cols = [
        "Sell Out Un. TT","MTD esperado 2026","MTD realizado 2026","Diff",
        "MTD realizado 2025","MTG necessário FLAT","MTG realizado em 2025"
    ]
    pct_cols = ["Evolução realizada","Evolução necessária PARA FLAT","Evolução prevista"]

    df_show = df_exec.copy()
    for c in num_cols:
        df_show[c] = df_show[c].apply(_fmt_number)
    for c in pct_cols:
        df_show[c] = df_show[c].apply(_fmt_percent)

    # Estilo: amarelo no FLAT, preto em Diff, negrito em principais
    styler = (df_show.style
        .apply(lambda col: ['background-color: #fff59d; font-weight: 700'
                            if col.name == "MTG necessário FLAT" and v != "" else '' for v in col], axis=0)
        .apply(lambda col: ['background-color: #212121; color: white; font-weight: 700'
                            if col.name == "Diff" and v != "" else '' for v in col], axis=0)
        .apply(lambda col: ['font-weight: 700'
                            if col.name in ["MTD esperado 2026", "MTD realizado 2026", "Evolução prevista"]
                            and v != "" else '' for v in col], axis=0)
    )

    st.dataframe(styler, use_container_width=True)

    # Download CSV do resumo
    st.download_button(
        "📥 Baixar Resumo Executivo (CSV)",
        data=df_exec.to_csv(index=False).encode("utf-8"),
        file_name=f"resumo_executivo_{sel_year}-{sel_month:02d}.csv",
        mime="text/csv"
    )

st.caption("💡 L'Oréal - Dashboard Sell-Out")
