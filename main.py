import os
import json
import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

# =========================
#   CONFIG DA PÁGINA
# =========================
st.set_page_config(page_title="Boost Sell-Out", layout="wide")

# =========================
#   LOGIN (JSON local robusto)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # se estiver corrompido, recria
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def create_account(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password  # (opcional) troque por hash seguro depois
    save_users(users)
    return True

def validate_login(username, password):
    users = load_users()
    return users.get(username) == password  # (opcional) troque por check de hash depois

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
    else:
        new_user = st.text_input("Novo usuário")
        new_pass = st.text_input("Nova senha", type="password")
        if st.button("Criar conta"):
            if not new_user or not new_pass:
                st.warning("Informe usuário e senha.")
            elif create_account(new_user, new_pass):
                st.success("Conta criada! Agora faça login.")
            else:
                st.error("Este usuário já existe.")

def require_login():
    if "auth" not in st.session_state or st.session_state["auth"] is not True:
        login_screen()
        st.stop()

# 🔒 Exigir login
require_login()

# =========================
#   FUNÇÕES DE DADOS E MODELO
# =========================
SEED = 42
np.random.seed(SEED)

def get_data_limite(reference_date=None):
    """Regra D-4 (sex a dom) / D-6 (seg a qui)"""
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    dia_semana = reference_date.dayofweek  # 0=seg ... 6=dom
    dias_atras = 4 if dia_semana >= 4 else 6
    data_limite = reference_date - pd.Timedelta(days=dias_atras)
    return data_limite, dias_atras

def read_boost_sheet(file_uploaded, sheet_name):
    """Lê a aba do template com colunas ['date','y']"""
    try:
        df = pd.read_excel(file_uploaded, sheet_name=sheet_name, skiprows=9, usecols='B:D')
        df.columns = ['Ano_Mes', 'Dia_Mes', 'Total']

        # Limpeza
        df = df.dropna(subset=['Ano_Mes', 'Dia_Mes'], how='all')
        df = df[~df['Ano_Mes'].astype(str).str.contains('Ano|Mês|Month', case=False, na=False)]

        df['Dia_Mes'] = pd.to_numeric(df['Dia_Mes'], errors='coerce')
        df = df.dropna(subset=['Dia_Mes'])

        df['Ano_Mes_str'] = df['Ano_Mes'].astype(str).str.strip()
        df['date'] = None
        try:
            df['date'] = pd.to_datetime(
                df['Dia_Mes'].astype(int).astype(str) + '-' + df['Ano_Mes_str'],
                format='%d-%m-%Y',
                errors='coerce'
            )
        except Exception:
            pass

        if df['date'].isna().all():
            try:
                ano_mes_date = pd.to_datetime(df['Ano_Mes'], errors='coerce')
                df['date'] = ano_mes_date + pd.to_timedelta(df['Dia_Mes'].astype(int) - 1, unit='D')
            except Exception:
                pass

        df = df.dropna(subset=['date'])

        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        df = df.dropna(subset=['Total'])

        df = df[['date', 'Total']].copy()
        df.columns = ['date', 'y']
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Erro ao ler aba {sheet_name}: {str(e)}")
        return None

# --------- CACHE: leitura por aba (usa bytes do arquivo) ---------
@st.cache_data(show_spinner=False, max_entries=20)
def read_boost_sheet_cached(file_bytes: bytes, sheet_name: str):
    bio = io.BytesIO(file_bytes)  # recria o arquivo em memória
    return read_boost_sheet(bio, sheet_name)

def build_exog(df, date_col='date'):
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
    X_lags = {f'lag_{i}': y.shift(i) for i in range(1, lags + 1)}
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
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            random_state=SEED,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1
        ))
    ])

    # ✅ Treina o modelo (esta linha é essencial)
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
        y_hat_clipped = max(y_hat, 0.0)  # sem negativos
        preds.append(y_hat_clipped)
        history.append(y_hat_clipped)

    return np.array(preds)

def cumulative(arr):
    return np.cumsum(np.asarray(arr))

def calcular_metricas_mtd(df_real, df_pred, data_limite, target_month_year):
    """
    Retorna métricas até a data limite (MTD) + total previsto do mês (FLAT previsto do app).
    """
    year, month = target_month_year
    mask1 = df_real['date'].dt.year == year
    mask2 = df_real['date'].dt.month == month
    mask3 = df_real['date'] <= data_limite

    mtd_realizado = df_real[mask1 & mask2 & mask3]['y'].sum()

    dia_atual = min(data_limite.day, len(df_pred))
    mtd_esperado = df_pred.iloc[:dia_atual]['y_pred'].sum()

    mtg_necessario = df_pred['y_pred'].sum()  # aqui no app original representa o total previsto do mês (FLAT)

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

# --------- CACHE: processamento por canal (treino + previsão) ---------
@st.cache_data(show_spinner=False, max_entries=100)
def processar_canal_cached(df_canal: pd.DataFrame,
                           data_limite_date: str,
                           sel_year: int, sel_month: int,
                           lags: int, last5_mult: float,
                           seed: int):
    data_limite = pd.to_datetime(data_limite_date)
    np.random.seed(seed)
    return processar_canal(df_canal, "CANAL", data_limite, sel_year, sel_month, lags, last5_mult)

# =========================
#   UI PRINCIPAL
# =========================
st.title("📊 Boost Sell-Out Dashboard")
st.caption("Análise dos canais OFICIAL, INDIRETO e DIRETO")

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

# ---- Leitura das abas (com cache por bytes) ----
with st.spinner("Lendo arquivo..."):
    file_bytes = up.getvalue()
    FILE_HASH = hashlib.md5(file_bytes).hexdigest()  # pode usar para diagnosticar/invalidar cache manualmente

    df_oficial  = read_boost_sheet_cached(file_bytes, 'DPGP (OFICIAL)')
    df_indireto = read_boost_sheet_cached(file_bytes, 'DPGP (INDIRETO)')
    df_direto   = read_boost_sheet_cached(file_bytes, 'DPGP (DIRETO)')

abas_disponiveis = []
if df_oficial is not None:  abas_disponiveis.append(("OFICIAL",  df_oficial))
if df_indireto is not None: abas_disponiveis.append(("INDIRETO", df_indireto))
if df_direto is not None:   abas_disponiveis.append(("DIRETO",   df_direto))

if not abas_disponiveis:
    st.error("Nenhuma aba lida")
    st.stop()

st.success(f"{len(abas_disponiveis)} canal(is) carregado(s)")

# ---- Mês-alvo ----
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

# ---- Processar canais (com cache) ----
dfs_by_name = {nome: df for (nome, df) in abas_disponiveis}
resultados = {}
for canal_name, df_canal in abas_disponiveis:
    resultados[canal_name] = processar_canal_cached(
        df_canal=df_canal,
        data_limite_date=str(pd.to_datetime(data_limite).date()),
        sel_year=sel_year,
        sel_month=sel_month,
        lags=lags,
        last5_mult=last5_multiplier,
        seed=SEED
    )

# ---- Tabs ----
tab_names  = [nome for nome, _ in abas_disponiveis]
tab_labels = [f"📈 {nome}" for nome in tab_names] + ["📑 Resultados Tabulados", "🧮 Simulador"]
tabs = st.tabs(tab_labels)

# ---- Abas por canal ----
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

# ============================
#   📑 Resultados Tabulados (quadros + ALERTAS/INSIGHTS)
# ============================
def _compute_baselines_2025(df_canal, sel_year, sel_month, data_limite):
    """Retorna (MTD_2025_alinhado, MTG_2025_cheio) para o mês-alvo."""
    if df_canal is None or df_canal.empty:
        return 0.0, 0.0
    year_25 = sel_year - 1
    last_day_25 = calendar.monthrange(year_25, sel_month)[1]
    corte = min(pd.to_datetime(data_limite).day, last_day_25)
    mask_mes_25 = (df_canal['date'].dt.year == year_25) & (df_canal['date'].dt.month == sel_month)
    mtd_25 = float(df_canal[mask_mes_25 & (df_canal['date'].dt.day <= corte)]['y'].sum())
    mtg_25 = float(df_canal[mask_mes_25]['y'].sum())
    return mtd_25, mtg_25

def _total_2025_mes_cheio(df_canal, sel_year, sel_month):
    """Total 2025 do mês cheio (para o canal)."""
    if df_canal is None or df_canal.empty:
        return 0.0
    year_25 = sel_year - 1
    mask = (df_canal['date'].dt.year == year_25) & (df_canal['date'].dt.month == sel_month)
    return float(df_canal[mask]['y'].sum())

def _fmt_num(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"

def _fmt_pct(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1f}%"

def _risk_level(evol_real, gap_pct, mtg_nec_units, mtg_25):
    """
    Engine de risco simples:
      - crítico: evolução <= -5% OU gap <= -10% OU MTG_nec > 120% do MTG_25
      - risco:   evolução < 0%  OU gap < 0%   OU MTG_nec > 100% do MTG_25
      - conforto: evolução >= +3% E gap >= +5% E MTG_nec <= 90% do MTG_25
      - atenção: demais casos
    """
    if mtg_25 is None or mtg_25 == 0:
        mtg_25 = 1e-9  # evita divisão por zero

    if (evol_real is not None and evol_real <= -5) or (gap_pct is not None and gap_pct <= -10) or (mtg_nec_units > 1.2 * mtg_25):
        return "critical"
    if (evol_real is not None and evol_real < 0) or (gap_pct is not None and gap_pct < 0) or (mtg_nec_units > 1.0 * mtg_25):
        return "risk"
    if (evol_real is not None and evol_real >= 3) and (gap_pct is not None and gap_pct >= 5) and (mtg_nec_units <= 0.9 * mtg_25):
        return "comfort"
    return "attention"

def _render_alert_block(titulo, evol_real, gap_pct, mtg_nec_units, mtg_25, evol_nec_flat):
    """Imprime um bloco de alerta estilo 3 (executivo) em função do nível de risco."""
    nivel = _risk_level(evol_real, gap_pct, mtg_nec_units, mtg_25)

    texto = (
        f"**{titulo}**\n\n"
        f"- Evolução realizada (MTD 2026 vs 2025): **{_fmt_pct(evol_real)}**\n"
        f"- Gap MTD (Realizado vs Previsto): **{_fmt_pct(gap_pct)}**\n"
        f"- MTG necessário p/ FLAT (unid.): **{_fmt_num(mtg_nec_units)}**\n"
        f"- MTG 2025 (histórico): **{_fmt_num(mtg_25)}**\n"
        f"- Evolução necessária p/ FLAT no MTG: **{_fmt_pct(evol_nec_flat)}**\n\n"
    )

    if nivel == "critical":
        st.error(
            texto +
            "🔴 **ALERTA CRÍTICO:** O canal apresenta **desaceleração relevante** no MTD e "
            "o MTG necessário para atingir ao menos 0% no mês está **acima da capacidade histórica**. "
            "Sem **aceleração imediata** no ritmo diário, o fechamento tende a ser **negativo**."
        )
    elif nivel == "risk":
        st.warning(
            texto +
            "🟠 **RISCO:** O canal está **abaixo do ritmo** e o MTG necessário supera o histórico. "
            "É possível recuperar, mas depende de **ganhos consistentes nos próximos dias**."
        )
    elif nivel == "attention":
        st.info(
            texto +
            "🟡 **ATENÇÃO:** O canal está próximo do **ponto de equilíbrio**. "
            "Monitorar o ritmo diário e ativar **alavancas táticas** pode garantir o FLAT."
        )
    else:  # comfort
        st.success(
            texto +
            "🟢 **CENÁRIO CONFORTÁVEL:** O desempenho está **acima do previsto** e o MTG necessário "
            "é **inferior** ao histórico. Tendência positiva de fechamento do mês."
        )

def _compute_kpis(nome, resultados, dfs_by_name, sel_year, sel_month, data_limite):
    """
    Retorna dicionário com:
      mtd_real_26, mtd_prev_26, mtd_25, evol_real_pct,
      total_25, mtg_25, mtg_nec_flat_units, evol_nec_flat_pct, gap_pct
    """
    if nome not in resultados or nome not in dfs_by_name or resultados[nome] is None:
        return None

    res = resultados[nome]
    df_c = dfs_by_name[nome]

    mtd_real_26 = res["metricas"]["mtd_realizado"]
    mtd_prev_26 = res["metricas"]["mtd_esperado"]
    mtd_25, mtg_25 = _compute_baselines_2025(df_c, sel_year, sel_month, data_limite)
    total_25 = _total_2025_mes_cheio(df_c, sel_year, sel_month)
    mtg_nec_flat_units = max(total_25 - mtd_real_26, 0.0)
    evol_real_pct = ((mtd_real_26 / mtd_25) - 1) * 100 if mtd_25 > 0 else np.nan
    evol_nec_flat_pct = ((mtg_nec_flat_units / mtg_25) - 1) * 100 if mtg_25 > 0 else np.nan
    gap_pct = ((mtd_real_26 - mtd_prev_26) / mtd_prev_26 * 100) if mtd_prev_26 > 0 else np.nan

    return {
        "nome": nome,
        "mtd_real_26": mtd_real_26,
        "mtd_prev_26": mtd_prev_26,
        "mtd_25": mtd_25,
        "evol_real_pct": evol_real_pct,
        "total_25": total_25,
        "mtg_25": mtg_25,
        "mtg_nec_flat_units": mtg_nec_flat_units,
        "evol_nec_flat_pct": evol_nec_flat_pct,
        "gap_pct": gap_pct
    }

with tabs[-2]:
    data_ref_str = pd.to_datetime(data_ref).strftime("%d/%m/%Y")
    data_lim_str = pd.to_datetime(data_limite).strftime("%d/%m/%Y")
    st.subheader(f"Cenário {data_ref_str} - prévia até {data_lim_str}")

    # Mês PT-BR
    meses_pt = {
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
        5: "maio", 6: "junho", 7: "julho", 8: "agosto",
        9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"
    }
    mes_25_titulo = f"{meses_pt.get(sel_month, '').capitalize()} {sel_year-1}"

    # Previsto mês cheio por canal
    pred_dir_26 = float(resultados["DIRETO"]["forecast_df"]["y_pred"].sum()) if "DIRETO" in resultados and resultados["DIRETO"] else None
    pred_ind_26 = float(resultados["INDIRETO"]["forecast_df"]["y_pred"].sum()) if "INDIRETO" in resultados and resultados["INDIRETO"] else None
    pred_dpgp_26 = (pred_dir_26 or 0.0) + (pred_ind_26 or 0.0) if (pred_dir_26 is not None and pred_ind_26 is not None) else None

    # Real 2025 (mês cheio)
    mtg_25_dir = _compute_baselines_2025(dfs_by_name.get("DIRETO"), sel_year, sel_month, data_limite)[1] if "DIRETO" in dfs_by_name else None
    mtg_25_ind = _compute_baselines_2025(dfs_by_name.get("INDIRETO"), sel_year, sel_month, data_limite)[1] if "INDIRETO" in dfs_by_name else None
    mtg_25_dpgp = (mtg_25_dir or 0.0) + (mtg_25_ind or 0.0) if (mtg_25_dir is not None and mtg_25_ind is not None) else None

    # % crescimento previsto (sem GAP)
    def _pc(val26, val25):
        if val26 is None or val25 in (None, 0.0):
            return np.nan
        return (val26 - val25) / val25 * 100.0

    cres_dir = _pc(pred_dir_26, mtg_25_dir)
    cres_ind = _pc(pred_ind_26, mtg_25_ind)
    cres_dpgp = _pc(pred_dpgp_26, mtg_25_dpgp)

    # ===== Quadros simples (sem HTML), bem destacados =====
    def quadro_unidades(titulo, valores_dict):
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            df_box = pd.DataFrame({
                "Direto": [valores_dict.get("Direto")],
                "Indireto": [valores_dict.get("Indireto")],
                "DPGP": [valores_dict.get("DPGP")],
            })
            st.dataframe(
                df_box.style.format("{:,.0f}"),
                use_container_width=True, hide_index=True
            )

    def quadro_percentual(titulo, valores_dict):
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            def _fmtp(v):
                return "" if (v is None or pd.isna(v)) else f"{v:+.1f}%"
            df_box = pd.DataFrame({
                "Direto": [_fmtp(valores_dict.get("Direto"))],
                "Indireto": [_fmtp(valores_dict.get("Indireto"))],
                "DPGP": [_fmtp(valores_dict.get("DPGP"))],
            })
            st.table(df_box)

    # ---- 3 quadros principais ----
    quadro_unidades("Sell Out Unidades PREVISTA", {
        "Direto": pred_dir_26, "Indireto": pred_ind_26, "DPGP": pred_dpgp_26
    })

    quadro_unidades(f"Sell Out Unidades {mes_25_titulo}", {
        "Direto": _total_2025_mes_cheio(dfs_by_name.get("DIRETO"), sel_year, sel_month) if "DIRETO" in dfs_by_name else None,
        "Indireto": _total_2025_mes_cheio(dfs_by_name.get("INDIRETO"), sel_year, sel_month) if "INDIRETO" in dfs_by_name else None,
        "DPGP": (
            (_total_2025_mes_cheio(dfs_by_name.get("DIRETO"), sel_year, sel_month) or 0.0) +
            (_total_2025_mes_cheio(dfs_by_name.get("INDIRETO"), sel_year, sel_month) or 0.0)
        ) if ("DIRETO" in dfs_by_name and "INDIRETO" in dfs_by_name) else None
    })

    quadro_percentual("%crescimento previsto modelo", {
        "Direto": cres_dir, "Indireto": cres_ind, "DPGP": cres_dpgp
    })

    # ===== Bloco GAP (contêiner maior para destacar) =====
    with st.container(border=True):
        st.subheader("🔻 Previsões Ajustadas com GAP")
        st.caption("O GAP é (MTD Previsto − MTD Realizado). As previsões abaixo descontam o GAP do mês cheio.")

        # GAP por canal
        def gap_acumulado(nome):
            res = resultados.get(nome)
            if not res:
                return None
            return res["metricas"]["mtd_esperado"] - res["metricas"]["mtd_realizado"]

        gap_dir = gap_acumulado("DIRETO")
        gap_ind = gap_acumulado("INDIRETO")
        gap_dpgp = (gap_dir or 0.0) + (gap_ind or 0.0) if (gap_dir is not None and gap_ind is not None) else None

        # Prevista GAP (mês cheio − gap acumulado)
        prev_gap_dir = (pred_dir_26 - gap_dir) if (pred_dir_26 is not None and gap_dir is not None) else None
        prev_gap_ind = (pred_ind_26 - gap_ind) if (pred_ind_26 is not None and gap_ind is not None) else None
        prev_gap_dpgp = (pred_dpgp_26 - gap_dpgp) if (pred_dpgp_26 is not None and gap_dpgp is not None) else None

        quadro_unidades("Sell Out Unidades PREVISTA GAP", {
            "Direto": prev_gap_dir, "Indireto": prev_gap_ind, "DPGP": prev_gap_dpgp
        })

        # % crescimento com GAP vs 2025
        cres_gap_dir = _pc(prev_gap_dir, _total_2025_mes_cheio(dfs_by_name.get("DIRETO"), sel_year, sel_month) if "DIRETO" in dfs_by_name else None)
        cres_gap_ind = _pc(prev_gap_ind, _total_2025_mes_cheio(dfs_by_name.get("INDIRETO"), sel_year, sel_month) if "INDIRETO" in dfs_by_name else None)
        cres_gap_dpgp = _pc(prev_gap_dpgp, (
            (_total_2025_mes_cheio(dfs_by_name.get("DIRETO"), sel_year, sel_month) or 0.0) +
            (_total_2025_mes_cheio(dfs_by_name.get("INDIRETO"), sel_year, sel_month) or 0.0)
        ) if ("DIRETO" in dfs_by_name and "INDIRETO" in dfs_by_name) else None)

        quadro_percentual("%crescimento previsto modelo GAP", {
            "Direto": cres_gap_dir, "Indireto": cres_gap_ind, "DPGP": cres_gap_dpgp
        })

    # ========= NOVO: PAINÉIS DE ALERTAS POR CANAL =========
    st.divider()
    st.subheader("🔔 Painéis de Alertas por Canal")

    kpi_dir = _compute_kpis("DIRETO", resultados, dfs_by_name, sel_year, sel_month, data_limite) if "DIRETO" in resultados else None
    kpi_ind = _compute_kpis("INDIRETO", resultados, dfs_by_name, sel_year, sel_month, data_limite) if "INDIRETO" in resultados else None

    # DPGP consolidado (somatório Direto + Indireto)
    kpi_dpgp = None
    if kpi_dir and kpi_ind:
        mtd_real_26 = (kpi_dir["mtd_real_26"] or 0.0) + (kpi_ind["mtd_real_26"] or 0.0)
        mtd_prev_26 = (kpi_dir["mtd_prev_26"] or 0.0) + (kpi_ind["mtd_prev_26"] or 0.0)
        mtd_25      = (kpi_dir["mtd_25"] or 0.0) + (kpi_ind["mtd_25"] or 0.0)
        total_25    = (kpi_dir["total_25"] or 0.0) + (kpi_ind["total_25"] or 0.0)
        mtg_25      = (kpi_dir["mtg_25"] or 0.0) + (kpi_ind["mtg_25"] or 0.0)

        evol_real = ((mtd_real_26 / mtd_25) - 1) * 100 if mtd_25 > 0 else np.nan
        gap_pct = ((mtd_real_26 - mtd_prev_26) / mtd_prev_26 * 100) if mtd_prev_26 > 0 else np.nan
        mtg_nec_flat_units = max(total_25 - mtd_real_26, 0.0)
        evol_nec_flat = ((mtg_nec_flat_units / mtg_25) - 1) * 100 if mtg_25 > 0 else np.nan

        kpi_dpgp = {
            "nome": "DPGP",
            "mtd_real_26": mtd_real_26,
            "mtd_prev_26": mtd_prev_26,
            "mtd_25": mtd_25,
            "evol_real_pct": evol_real,
            "total_25": total_25,
            "mtg_25": mtg_25,
            "mtg_nec_flat_units": mtg_nec_flat_units,
            "evol_nec_flat_pct": evol_nec_flat,
            "gap_pct": gap_pct
        }

    # Render por canal
    colA, colB = st.columns(2)
    if kpi_dir:
        with colA:
            _render_alert_block(
                "DIRETO — Alerta Executivo",
                kpi_dir["evol_real_pct"], kpi_dir["gap_pct"],
                kpi_dir["mtg_nec_flat_units"], kpi_dir["mtg_25"], kpi_dir["evol_nec_flat_pct"]
            )
    if kpi_ind:
        with colB:
            _render_alert_block(
                "INDIRETO — Alerta Executivo",
                kpi_ind["evol_real_pct"], kpi_ind["gap_pct"],
                kpi_ind["mtg_nec_flat_units"], kpi_ind["mtg_25"], kpi_ind["evol_nec_flat_pct"]
            )

    if kpi_dpgp:
        st.markdown(" ")
        _render_alert_block(
            "DPGP — Alerta Executivo (Consolidado Direto + Indireto)",
            kpi_dpgp["evol_real_pct"], kpi_dpgp["gap_pct"],
            kpi_dpgp["mtg_nec_flat_units"], kpi_dpgp["mtg_25"], kpi_dpgp["evol_nec_flat_pct"]
        )

    # ========= NOVO: PAINEL EXECUTIVO CONSOLIDADO =========
    with st.container(border=True):
        st.subheader("🧭 Painel Executivo Consolidado (Resumo do Mês)")
        bullets = []
        def add_bullet(txt): bullets.append(f"- {txt}")

        if kpi_dir:
            add_bullet(
                f"**Direto** — Evolução MTD: {_fmt_pct(kpi_dir['evol_real_pct'])}, "
                f"Gap MTD vs Prev.: {_fmt_pct(kpi_dir['gap_pct'])}, "
                f"MTG necessário: {_fmt_num(kpi_dir['mtg_nec_flat_units'])} "
                f"(hist. 2025: {_fmt_num(kpi_dir['mtg_25'])})."
            )
        if kpi_ind:
            add_bullet(
                f"**Indireto** — Evolução MTD: {_fmt_pct(kpi_ind['evol_real_pct'])}, "
                f"Gap MTD vs Prev.: {_fmt_pct(kpi_ind['gap_pct'])}, "
                f"MTG necessário: {_fmt_num(kpi_ind['mtg_nec_flat_units'])} "
                f"(hist. 2025: {_fmt_num(kpi_ind['mtg_25'])})."
            )
        if kpi_dpgp:
            add_bullet(
                f"**DPGP** — Evolução MTD: {_fmt_pct(kpi_dpgp['evol_real_pct'])}, "
                f"Gap MTD vs Prev.: {_fmt_pct(kpi_dpgp['gap_pct'])}, "
                f"MTG necessário: {_fmt_num(kpi_dpgp['mtg_nec_flat_units'])} "
                f"(hist. 2025: {_fmt_num(kpi_dpgp['mtg_25'])})."
            )

        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.info("Sem dados suficientes para consolidar o painel executivo.")

# ============================
#   🧮 Simulador – baseado em % de evolução
# ============================
with tabs[-1]:
    st.header("🧮 Simulador de Evolução (%) por Canal")

    # ========= BASELINE 2025 =========
    def _compute_baselines_2025_local(df_canal, sel_year, sel_month, data_limite):
        if df_canal is None or df_canal.empty:
            return 0.0, 0.0
        year_25 = sel_year - 1
        last_day_25 = calendar.monthrange(year_25, sel_month)[1]
        corte = min(pd.to_datetime(data_limite).day, last_day_25)
        mask_mes_25 = (df_canal["date"].dt.year == year_25) & (df_canal["date"].dt.month == sel_month)
        mtd_25 = float(df_canal[mask_mes_25 & (df_canal["date"].dt.day <= corte)]["y"].sum())
        mtg_25 = float(df_canal[mask_mes_25]["y"].sum())
        return mtd_25, mtg_25

    # Real 2025 (mês cheio e MTD)
    mtd25_dir, total25_dir = _compute_baselines_2025_local(dfs_by_name.get("DIRETO"), sel_year, sel_month, data_limite) if "DIRETO" in dfs_by_name else (0.0, 0.0)
    mtd25_ind, total25_ind = _compute_baselines_2025_local(dfs_by_name.get("INDIRETO"), sel_year, sel_month, data_limite) if "INDIRETO" in dfs_by_name else (0.0, 0.0)
    total25_dpgp = total25_dir + total25_ind
    mtg25_dpgp = (total25_dir - mtd25_dir) + (total25_ind - mtd25_ind)

    st.caption("Simule diretamente **% evolução por canal** e veja o impacto no DPGP e nas unidades acumuladas.")

    # ========= INPUTS DO USUÁRIO (formulário para evitar rerun a cada tecla) =========
    with st.form("sim_pct_form"):
        c1, c2 = st.columns(2)
        evo_dir_pct = c1.number_input("% evolução Direto", value=0.0, step=0.1, format="%.1f")
        evo_ind_pct = c2.number_input("% evolução Indireto", value=0.0, step=0.1, format="%.1f")
        submitted = st.form_submit_button("Calcular")

    # Persistência simples do último cenário (não relê enquanto não clicar)
    if (not submitted) and ("sim_last" in st.session_state):
        evo_dir_pct = st.session_state["sim_last"]["evo_dir_pct"]
        evo_ind_pct = st.session_state["sim_last"]["evo_ind_pct"]
    else:
        st.session_state["sim_last"] = {"evo_dir_pct": evo_dir_pct, "evo_ind_pct": evo_ind_pct}

    # Converter % → fração
    evo_dir = evo_dir_pct / 100.0
    evo_ind = evo_ind_pct / 100.0

    # ========= CÁLCULOS =========
    # Totais simulados (mês cheio) por canal
    sim_dir_units = total25_dir * (1.0 + evo_dir) if total25_dir > 0 else 0.0
    sim_ind_units = total25_ind * (1.0 + evo_ind) if total25_ind > 0 else 0.0
    sim_dpgp_units = sim_dir_units + sim_ind_units

    # MTD simulado (até a data) por canal
    sim_mtd_dir = mtd25_dir * (1.0 + evo_dir) if mtd25_dir > 0 else 0.0
    sim_mtd_ind = mtd25_ind * (1.0 + evo_ind) if mtd25_ind > 0 else 0.0
    sim_mtd_dpgp = sim_mtd_dir + sim_mtd_ind

    # % Evolução simulada DPGP vs 2025 (mês cheio)
    evo_dpgp_pct = ((sim_dpgp_units - total25_dpgp) / total25_dpgp * 100.0) if total25_dpgp > 0 else None

    # Pressão do MTG (sim): quanto falta para FLAT, dado o MTD simulado
    mtg_nec_sim = max(total25_dpgp - sim_mtd_dpgp, 0.0)
    evol_nec_sim_pct = ((mtg_nec_sim / mtg25_dpgp) - 1) * 100.0 if mtg25_dpgp > 0 else None

    # ========= MÉTRICAS =========
    c1, c2, c3 = st.columns(3)
    c1.metric("Direto (simulado - mês cheio)", f"{sim_dir_units:,.0f}", delta=f"{evo_dir_pct:+.1f}%")
    c2.metric("Indireto (simulado - mês cheio)", f"{sim_ind_units:,.0f}", delta=f"{evo_ind_pct:+.1f}%")
    c3.metric("DPGP (simulado - mês cheio)", f"{sim_dpgp_units:,.0f}", delta=f"{(evo_dpgp_pct if evo_dpgp_pct is not None else 0):+.1f}% vs 2025" if evo_dpgp_pct is not None else "N/D")

    st.divider()

    # ========= TABELAS =========
    with st.container(border=True):
        st.subheader("📦 Sell Out Unidades — Simulado (mês cheio)")
        st.dataframe(
            pd.DataFrame({
                "Direto": [sim_dir_units],
                "Indireto": [sim_ind_units],
                "DPGP": [sim_dpgp_units]
            }).style.format("{:,.0f}"),
            hide_index=True,
            use_container_width=True
        )

    with st.container(border=True):
        st.subheader("📈 % Crescimento — Simulado (mês cheio)")
        st.table(
            pd.DataFrame({
                "Direto": [f"{evo_dir_pct:+.1f}%"],
                "Indireto": [f"{evo_ind_pct:+.1f}%"],
                "DPGP": [f"{evo_dpgp_pct:+.1f}%"] if evo_dpgp_pct is not None else ["N/D"]
            })
        )

    # ========= DOWNLOAD =========
    csv_sim = pd.DataFrame({
        "Canal": ["Direto", "Indireto", "DPGP"],
        "Unidades Simuladas (mês cheio)": [sim_dir_units, sim_ind_units, sim_dpgp_units],
        "% Evolução (mês cheio)": [f"{evo_dir_pct:+.1f}%",
                                   f"{evo_ind_pct:+.1f}%",
                                   f"{evo_dpgp_pct:+.1f}%" if evo_dpgp_pct is not None else "N/D"]
    })
    st.download_button(
        "📥 Baixar Simulação (%) (CSV)",
        data=csv_sim.to_csv(index=False).encode("utf-8"),
        file_name=f"simulador_pct_{sel_year}-{sel_month:02d}.csv",
        mime="text/csv"
    )

    # ========= NOVO: CARD EXECUTIVO DO CENÁRIO SIMULADO =========
    with st.container(border=True):
        st.subheader("🧭 Resumo Executivo do Cenário Simulado")
        # Risk engine simples para o simulado, usando mtg_nec_sim vs histórico e evolução simulada
        def sim_risk(evo_pct, mtg_nec_sim_u, mtg25_u):
            if mtg25_u <= 0:
                mtg25_u = 1e-9
            if evo_pct is not None and (evo_pct <= -5 or mtg_nec_sim_u > 1.2 * mtg25_u):
                return "critical"
            if evo_pct is not None and (evo_pct < 0 or mtg_nec_sim_u > 1.0 * mtg25_u):
                return "risk"
            if evo_pct is not None and (evo_pct >= 3) and (mtg_nec_sim_u <= 0.9 * mtg25_u):
                return "comfort"
            return "attention"

        nivel_sim = sim_risk(evo_dpgp_pct, mtg_nec_sim, mtg25_dpgp)

        texto = (
            f"- **DPGP (mês cheio simulado)**: **{sim_dpgp_units:,.0f}** "
            f"({f'{evo_dpgp_pct:+.1f}%' if evo_dpgp_pct is not None else 'N/D'} vs 2025)\n"
            f"- **MTD simulado até a data (DPGP)**: **{sim_mtd_dpgp:,.0f}**\n"
            f"- **MTG necessário p/ FLAT (sim)**: **{mtg_nec_sim:,.0f}** "
            f"(hist. 2025 MTG: **{mtg25_dpgp:,.0f}**, necessidade: {f'{evol_nec_sim_pct:+.1f}%' if evol_nec_sim_pct is not None else 'N/D'})\n"
            f"- **Contribuição**: Direto {sim_dir_units:,.0f} | Indireto {sim_ind_units:,.0f}\n"
        )

        if nivel_sim == "critical":
            st.error(
                texto + "\n" +
                "🔴 **ALERTA CRÍTICO | SIMULADO:** Mesmo no cenário proposto, o MTG necessário "
                "permanece **muito acima** do histórico e a evolução projetada é **desfavorável**. "
                "Serão necessárias **alavancas fortes** no curto prazo."
            )
        elif nivel_sim == "risk":
            st.warning(
                texto + "\n" +
                "🟠 **RISCO | SIMULADO:** O cenário melhora parcialmente, mas ainda exige "
                "**aceleração acima do histórico** no restante do mês para garantir o FLAT."
            )
        elif nivel_sim == "attention":
            st.info(
                texto + "\n" +
                "🟡 **ATENÇÃO | SIMULADO:** O cenário projetado está **próximo do equilíbrio**. "
                "A execução dos próximos dias será determinante para assegurar o FLAT."
            )
        else:
            st.success(
                texto + "\n" +
                "🟢 **CONFORTO | SIMULADO:** O cenário indica **probabilidade alta** de fechamento "
                "no FLAT ou acima, com **pressão reduzida** sobre o MTG."
            )

# Rodapé
st.caption("💡 L'Oréal - Dashboard Sell-Out")
