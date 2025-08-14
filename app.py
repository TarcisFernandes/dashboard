# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

st.set_page_config(
    page_title="Analisador de CSV com IA",
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
    
    # Tenta identificar colunas de data que podem ter sido lidas como 'object'
    potential_date_cols = []
    for col in categorical_cols:
        try:
            # Tenta converter a coluna para datetime sem gerar erros na maioria das linhas
            if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.8 * len(df):
                potential_date_cols.append(col)
        except (TypeError, ValueError):
            continue
            
    # Remove as colunas de data da lista de categóricas
    categorical_cols = [col for col in categorical_cols if col not in potential_date_cols]
    
    return numeric_cols, categorical_cols, potential_date_cols

# --- SIDEBAR E LÓGICA DE UPLOAD ---

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader(
    "Faça o upload do seu arquivo CSV",
    type=['csv']
)

# A LÓGICA PRINCIPAL DO APP AGORA DEPENDE DO UPLOAD
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Classifica as colunas do DataFrame enviado
    numeric_cols, categorical_cols, date_cols = classificar_colunas(df)
    
    # Converte colunas de data identificadas
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # --- GERAÇÃO DINÂMICA DE FILTROS ---
    st.sidebar.markdown("---")
    st.sidebar.title("Filtros Dinâmicos")
    
    # Cria um dicionário para armazenar os filtros aplicados
    filtros = {}
    
    # Cria filtros para colunas categóricas e de data
    colunas_filtragem = categorical_cols + date_cols
    for col in colunas_filtragem:
        if col in date_cols:
            # Filtro de intervalo de datas
            min_date = df[col].min()
            max_date = df[col].max()
            filtro_data = st.sidebar.date_input(f"Filtro para {col}", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(filtro_data) == 2:
                filtros[col] = filtro_data
        else:
            # Filtro de múltipla seleção
            opcoes = df[col].unique()
            selecao = st.sidebar.multiselect(f"Filtro para {col}", options=opcoes, default=opcoes)
            filtros[col] = selecao

    # Aplica os filtros ao DataFrame
    df_filtrado = df.copy()
    for col, valores in filtros.items():
        if col in date_cols:
            df_filtrado = df_filtrado[(df_filtrado[col] >= pd.to_datetime(valores[0])) & (df_filtrado[col] <= pd.to_datetime(valores[1]))]
        else:
            df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]


    # --- LAYOUT DA DASHBOARD DINÂMICA ---

    st.title("🚀 Dashboard Analítica Gerada por IA")
    st.markdown(f"Análise do arquivo: `{uploaded_file.name}`")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Dinâmica", "🤖 Análise com IA", "🔍 Visualizar Dados"])

    # --- ABA 1: DASHBOARD DINÂMICA ---
    with tab1:
        st.header("Indicadores Chave (KPIs)")
        if not numeric_cols:
            st.warning("Nenhuma coluna numérica encontrada para gerar KPIs.")
        else:
            # Permite ao usuário escolher qual coluna numérica analisar
            coluna_kpi = st.selectbox("Selecione a coluna numérica para os KPIs:", numeric_cols)
            if coluna_kpi:
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
            
            if col_cat_grafico and col_num_grafico:
                agg_df = df_filtrado.groupby(col_cat_grafico)[col_num_grafico].sum().reset_index()
                fig = px.bar(agg_df, x=col_cat_grafico, y=col_num_grafico, title=f"{col_num_grafico} por {col_cat_grafico}")
                st.plotly_chart(fig, use_container_width=True)

    # --- ABA 2: ANÁLISE COM IA ---
    with tab2:
        st.header("Análise Qualitativa com Gemini")
        if st.button("Gerar Análise dos Dados Filtrados", disabled=not GEMINI_CONFIGURADO):
            with st.spinner("Gemini está analisando seus dados... 🧠"):
                prompt = f"""
                Você é um analista de dados. Analise o seguinte conjunto de dados de um arquivo CSV.
                O resumo estatístico das colunas numéricas é (em formato JSON):
                {df_filtrado[numeric_cols].describe().to_json()}

                As 5 primeiras linhas dos dados são:
                {df_filtrado.head().to_string()}

                Com base nesses dados, escreva uma análise geral, identificando possíveis insights,
                tendências ou pontos de atenção. Seja claro e direto.
                """
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                st.subheader("Análise Gerada por IA:")
                st.markdown(response.text)

    # --- ABA 3: VISUALIZAR DADOS ---
    with tab3:
        st.header("Visualização dos Dados Filtrados")
        st.dataframe(df_filtrado)


# --- TELA INICIAL ANTES DO UPLOAD ---
else:
    st.info("👋 Bem-vindo ao Analisador de CSV com IA! Por favor, faça o upload de um arquivo CSV para começar.")