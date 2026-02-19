import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="Boost Sell-Out", layout="wide")

SEED = 42

def get_data_limite(reference_date=None):
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    dia_semana = reference_date.dayofweek
    dias_atras = 4 if dia_semana >= 4 else 6
    data_limite = reference_date - pd.Timedelta(days=dias_atras)
    return data_limite, dias_atras

def read_boost_sheet(file_uploaded, sheet_name):
    try:
        df = pd.read_excel(file_uploaded, sheet_name=sheet_name, skiprows=9, usecols='B:D')
        df.columns = ['Ano_Mes', 'Dia_Mes', 'Total']
        df = df.dropna(subset=['Ano_Mes', 'Dia_Mes'], how='all')
        df = df[~df['Ano_Mes'].astype(str).str.contains('Ano|Mês|Month', case=False, na=False)]
        df['Dia_Mes'] = pd.to_numeric(df['Dia_Mes'], errors='coerce')
        df = df.dropna(subset=['Dia_Mes'])
        df['Ano_Mes_str'] = df['Ano_Mes'].astype(str).str.strip()
        df['date'] = None
        try:
            df['date'] = pd.to_datetime(df['Dia_Mes'].astype(int).astype(str) + '-' + df['Ano_Mes_str'], format='%d-%m-%Y', errors='coerce')
        except:
            pass
        if df['date'].isna().all():
            try:
                ano_mes_date = pd.to_datetime(df['Ano_Mes'], errors='coerce')
                df['date'] = ano_mes_date + pd.to_timedelta(df['Dia_Mes'].astype(int) - 1, unit='D')
            except:
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
    out['is_bday'] = (out['dow'] <5).astype(int)
    out['month_progress'] = out['day'] / month_end_day
    return out

def supervised_from_series(y, exog_df, lags=14):
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
    X_hist, y_hist = supervised_from_series(y_series, exog_hist.iloc[:len(y_series)], lags=lags)
    num_cols = X_hist.columns.tolist()
    pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', random_state=SEED, max_iter=2000))
    ])
    pipe.fit(X_hist[num_cols], y_hist)
    history = list(y_series.values)
    preds = []
    for h in range(horizon):
        ex_h = exog_future.iloc[h:h+1].reset_index(drop=True)
        last_vals = history[-lags:][::-1]
        if len(last_vals) <lags:
            last_vals = ([history[-1]] * (lags - len(last_vals))) + last_vals
        row = {f'lag_{i+1}': last_vals[i] for i in range(lags)}
        row_df = pd.DataFrame([row])
        row_df = pd.concat([row_df, ex_h], axis=1)
        row_df = row_df.reindex(columns=num_cols, fill_value=0)
        y_hat = float(pipe.predict(row_df)[0])
        preds.append(max(y_hat, 0.0))
        history.append(y_hat)
    return np.array(preds)

def cumulative(arr):
    return np.cumsum(np.asarray(arr))

def calcular_metricas_mtd(df_real, df_pred, data_limite, target_month_year):
    year, month = target_month_year
    mask1 = df_real['date'].dt.year == year
    mask2 = df_real['date'].dt.month == month
    mask3 = df_real['date'] <= data_limite
    mtd_realizado = df_real[mask1 & mask2 & mask3]['y'].sum()
    dia_atual = min(data_limite.day, len(df_pred))
    mtd_esperado = df_pred.iloc[:dia_atual]['y_pred'].sum()
    mtg_necessario = df_pred['y_pred'].sum()
    diff = mtd_realizado - mtd_esperado
    evolucao = (diff / mtd_esperado * 100) if mtd_esperado >0 else 0
    return {
        'mtd_esperado': mtd_esperado,
        'mtd_realizado': mtd_realizado,
        'mtg_necessario': mtg_necessario,
        'diff': diff,
        'evolucao': evolucao
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
    preds = fit_predict_mlp(y_series=y_series, exog_hist=hist_exog[exog_cols], exog_future=fut_exog_full[exog_cols], horizon=horizon, lags=lags)
    forecast_df = pd.DataFrame({'date': fut_dates, 'y_pred': preds})
    mask1 = df['date'].dt.year == sel_year
    mask2 = df['date'].dt.month == sel_month
    mask3 = df['date'] <= data_limite
    real_month = df[mask1 & mask2 & mask3].copy().sort_values('date')
    metricas = calcular_metricas_mtd(df, forecast_df, data_limite, (sel_year, sel_month))
    return {'forecast_df': forecast_df, 'real_month': real_month, 'metricas': metricas}

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

with st.spinner("Lendo arquivo..."):
    df_oficial = read_boost_sheet(up, 'DPGP (OFICIAL)')
    df_indireto = read_boost_sheet(up, 'DPGP (INDIRETO)')
    df_direto = read_boost_sheet(up, 'DPGP (DIRETO)')

abas_disponiveis = []
if df_oficial is not None:
    abas_disponiveis.append(("OFICIAL", df_oficial))
if df_indireto is not None:
    abas_disponiveis.append(("INDIRETO", df_indireto))
if df_direto is not None:
    abas_disponiveis.append(("DIRETO", df_direto))

if not abas_disponiveis:
    st.error("Nenhuma aba lida")
    st.stop()

st.success(f"{len(abas_disponiveis)} canal(is) carregado(s)")

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

tab_names = [nome for nome, _ in abas_disponiveis]
tabs = st.tabs([f"📈 {nome}" for nome in tab_names])

for idx, (canal_name, df_canal) in enumerate(abas_disponiveis):
    with tabs[idx]:
        st.header(f"Canal: {canal_name}")
        resultado = processar_canal(df_canal, canal_name, data_limite, sel_year, sel_month, lags, last5_multiplier)
        if resultado is None:
            continue
        forecast_df = resultado['forecast_df']
        real_month = resultado['real_month']
        metricas = resultado['metricas']
        
        st.subheader("📊 Métricas")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MTD Esperado", f"{metricas['mtd_esperado']:,.0f}")
        with col2:
            st.metric("MTD Realizado", f"{metricas['mtd_realizado']:,.0f}", delta=f"{metricas['evolucao']:+.1f}%")
        with col3:
            st.metric("Diferença", f"{metricas['diff']:+,.0f}")
        with col4:
            st.metric("MTG Necessário", f"{metricas['mtg_necessario']:,.0f}")
        
        st.subheader("📈 Acumulado")
        fig1 = go.Figure()
        pred_cum = cumulative(forecast_df['y_pred'].values)
        fig1.add_trace(go.Scatter(x=forecast_df['date'], y=pred_cum, mode='lines+markers', name='Previsto', line=dict(color='#4A74F3', width=3)))
        if not real_month.empty:
            real_cum = cumulative(real_month['y'].values)
            fig1.add_trace(go.Scatter(x=real_month['date'], y=real_cum, mode='lines+markers', name='Realizado', line=dict(color='#9D5CE6', width=3)))
        fig1.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("📊 Diário")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=forecast_df['date'], y=forecast_df['y_pred'], name='Previsto', marker_color='#6C8BE0'))
        if not real_month.empty:
            fig2.add_trace(go.Scatter(x=real_month['date'], y=real_month['y'], name='Realizado', mode='lines+markers', line=dict(color='#B55CE6', width=3)))
        fig2.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("📋 Dados"):
            output_df = forecast_df.copy()
            output_df['pred_cum'] = pred_cum
            if not real_month.empty:
                output_df = output_df.merge(real_month[['date', 'y']].rename(columns={'y': 'real'}), on='date', how='left')
                output_df['real_cum'] = output_df['real'].cumsum()
            output_df['date'] = output_df['date'].dt.strftime('%d/%m/%Y')
            st.dataframe(output_df, use_container_width=True)
        
        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Baixar {canal_name}", data=csv_data, file_name=f"previsao_{canal_name}.csv")

if len(abas_disponiveis) >1:
    st.divider()
    st.header("🔄 Comparativo")
    fig_comp = go.Figure()
    for canal_name, df_canal in abas_disponiveis:
        resultado = processar_canal(df_canal, canal_name, data_limite, sel_year, sel_month, lags, last5_multiplier)
        if resultado:
            pred_cum = cumulative(resultado['forecast_df']['y_pred'].values)
            fig_comp.add_trace(go.Scatter(x=resultado['forecast_df']['date'], y=pred_cum, mode='lines+markers', name=canal_name))
    fig_comp.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig_comp, use_container_width=True)

st.caption("💡 L'Oréal - Dashboard Sell-Out")
