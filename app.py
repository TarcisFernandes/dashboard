# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import camelot
import os

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

st.set_page_config(
    page_title="Analisador de Documentos com IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_CONFIGURADO = True
except Exception as e:
    st.error(f"Erro ao configurar a API do Gemini. Verifique a chave 'GEMINI_API_KEY' nos segredos. Erro: {e}")
    GEMINI_CONFIGURADO = False

# --- FUNÇÕES AUXILIARES ---

@st.cache_data
def classificar_colunas(df):
    """Analisa o DataFrame e classifica as colunas em numéricas, categóricas e de data."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    potential_date_cols = []
    categorical_cols_copy = categorical_cols[:]
    
    for col in categorical_cols_copy:
        try:
            temp_col = pd.to_datetime(df[col], errors='coerce')
            if temp_col.notna().sum() / len(df) > 0.8:
                potential_date_cols.append(col)
        except (TypeError, ValueError):
            continue
            
    categorical_cols = [col for col in categorical_cols if col not in potential_date_cols]
    
    return numeric_cols, categorical_cols, potential_date_cols

# --- SIDEBAR E LÓGICA DE UPLOAD ---

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader(
    "Faça o upload do seu arquivo (CSV ou PDF)",
    type=['csv', 'pdf']
)

df = None

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)

    elif file_extension == 'pdf':
        with st.spinner("Lendo tabelas do PDF... Isso pode levar um momento."):
            try:
                temp_pdf_path = f"./temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='stream')
                os.remove(temp_pdf_path)
                
                if len(tables) > 0:
                    st.sidebar.success(f"{len(tables)} segmento(s) de tabela encontrado(s) e combinado(s).")
                    
                    cleaned_dfs = []
                    for table in tables:
                        temp_df = table.df
                        if not temp_df.empty:
                            new_header = temp_df.iloc[0]
                            temp_df = temp_df[1:]
                            temp_df.columns = new_header
                            cleaned_dfs.append(temp_df)
                    
                    if cleaned_dfs:
                        df = pd.concat(cleaned_dfs, ignore_index=True)
                        st.sidebar.info(f"O documento final tem {len(df)} linhas.")
                    else:
                        st.error("As tabelas encontradas no PDF estavam vazias.")
                else:
                    st.error("Nenhuma tabela foi encontrada neste arquivo PDF.")
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o PDF: {e}")

# --- PROCESSAMENTO DO DATAFRAME E GERAÇÃO DA DASHBOARD ---
if df is not None:
    df = df.rename(columns=lambda x: str(x).strip())
    
    numeric_cols, categorical_cols, date_cols = classificar_colunas(df)
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(how='all', inplace=True)

    st.sidebar.markdown("---")
    st.sidebar.title("Filtros Dinâmicos")
    
    filtros = {}
    colunas_filtragem = categorical_cols + date_cols
    for col in colunas_filtragem:
        if col in date_cols and not df[col].isnull().all():
            min_date, max_date = df[col].min(), df[col].max()
            filtro_data = st.sidebar.date_input(f"Filtro para {col}", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(filtro_data) == 2:
                filtros[col] = filtro_data
        elif col in categorical_cols:
            opcoes = df[col].dropna().unique()
            selecao = st.sidebar.multiselect(f"Filtro para {col}", options=opcoes, default=opcoes)
            filtros[col] = selecao

    df_filtrado = df.copy()
    for col, valores in filtros.items():
        if col in date_cols and pd.api.types.is_datetime64_any_dtype(df_filtrado[col]):
            df_filtrado = df_filtrado[(df_filtrado[col].dt.date >= valores[0]) & (df_filtrado[col].dt.date <= valores[1])]
        else:
            df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

    st.title("🚀 Dashboard Analítica Gerada por IA")
    st.markdown(f"Análise do arquivo: `{uploaded_file.name}`")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Dinâmica", "🤖 Análise com IA", "🔍 Visualizar Dados"])

    with tab1:
        st.header("Indicadores Chave (KPIs)")
        if not numeric_cols:
            st.warning("Nenhuma coluna numérica encontrada para gerar KPIs.")
        else:
            coluna_kpi = st.selectbox("Selecione a coluna numérica para os KPIs:", numeric_cols)
            if coluna_kpi and not df_filtrado.empty:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"Soma de {coluna_kpi}", f"{df_filtrado[coluna_kpi].sum():,.2f}")
                col2.metric(f"Média de {coluna_kpi}", f"{df_filtrado[coluna_kpi].mean():,.2f}")
                col3.metric(f"Valor Máximo", f"{df_filtrado[coluna_kpi].max():,.2f}")
                col4.metric(f"Contagem de Registros", df_filtrado.shape[0])
        
        st.markdown("---")
        st.header("Visualização Gráfica")
        if not categorical_cols or not numeric_cols:
            st.warning("É necessário ter ao menos uma coluna categórica e uma numérica para gerar gráficos.")
        else:
            col_cat_grafico = st.selectbox("Selecione a coluna para o Eixo X (Categórica):", categorical_cols)
            col_num_grafico = st.selectbox("Selecione a coluna para o Eixo Y (Numérica):", numeric_cols)
            
            if col_cat_grafico and col_num_grafico and not df_filtrado.empty:
                agg_df = df_filtrado.groupby(col_cat_grafico, as_index=False)[col_num_grafico].sum()
                fig = px.bar(agg_df, x=col_cat_grafico, y=col_num_grafico, title=f"{col_num_grafico} por {col_cat_grafico}")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Análise Qualitativa com Gemini")
        if df_filtrado.empty:
            st.warning("Não há dados para analisar com os filtros selecionados.")
        elif not numeric_cols:
            st.warning("Não há colunas numéricas para gerar análise estatística.")
        else:
            if st.button("Gerar Análise dos Dados Filtrados", disabled=not GEMINI_CONFIGURADO):
                with st.spinner("Gemini está analisando seus dados... 🧠"):
                    prompt = f"""
                    Você é um analista de dados. Analise o seguinte conjunto de dados.
                    Resumo estatístico (JSON): {df_filtrado[numeric_cols].describe().to_json()}
                    5 primeiras linhas: {df_filtrado.head().to_string()}
                    Com base nisso, escreva uma análise geral com insights, tendências ou pontos de atenção.
                    """
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt)
                    st.subheader("Análise Gerada por IA:")
                    st.markdown(response.text)

    with tab3:
        st.header("Visualização dos Dados Filtrados")
        st.dataframe(df_filtrado)

else:
    st.info("👋 Bem-vindo! Por favor, faça o upload de um arquivo CSV ou PDF para começar.")