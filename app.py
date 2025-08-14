# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

# Configura a página do Streamlit
st.set_page_config(
    page_title="Dashboard de Vendas com IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configura a API do Gemini usando os segredos do Streamlit
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_CONFIGURADO = True
except Exception as e:
    st.error(f"Erro ao configurar a API do Gemini. Verifique se a chave 'GEMINI_API_KEY' está nos segredos do seu app no Streamlit Cloud. Erro: {e}")
    GEMINI_CONFIGURADO = False


# --- 2. FUNÇÕES AUXILIARES ---

@st.cache_data # Cache para melhorar a performance
def carregar_dados(caminho_arquivo):
    """Carrega os dados de um arquivo CSV. Se o arquivo não existir, cria um com dados de exemplo."""
    if not os.path.exists(caminho_arquivo):
        st.warning(f"Arquivo '{caminho_arquivo}' não encontrado! Criando um com dados de exemplo.")
        dados_exemplo = {
            'Data': pd.to_datetime(['2025-08-04', '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-11', '2025-08-12', '2025-08-13']),
            'Produto': ['Produto A', 'Produto B', 'Produto A', 'Produto C', 'Produto B', 'Produto A', 'Produto C', 'Produto B'],
            'Categoria': ['Eletrônicos', 'Livros', 'Eletrônicos', 'Casa', 'Livros', 'Eletrônicos', 'Casa', 'Livros'],
            'Vendas': [1200, 150, 1800, 300, 250, 2200, 450, 210],
            'Regiao': ['Sudeste', 'Sul', 'Sudeste', 'Nordeste', 'Sul', 'Sudeste', 'Nordeste', 'Sul']
        }
        df = pd.DataFrame(dados_exemplo)
        df.to_csv(caminho_arquivo, index=False)
    
    df = pd.read_csv(caminho_arquivo)
    df['Data'] = pd.to_datetime(df['Data'])
    return df

# --- 3. CARREGAMENTO DOS DADOS E FILTROS ---

# Carrega os dados
df = carregar_dados('vendas.csv')

st.sidebar.title("Filtros Interativos")
regiao = st.sidebar.multiselect(
    "Selecione a Região:",
    options=df["Regiao"].unique(),
    default=df["Regiao"].unique()
)
categoria = st.sidebar.multiselect(
    "Selecione a Categoria:",
    options=df["Categoria"].unique(),
    default=df["Categoria"].unique()
)
df_filtrado = df[df["Regiao"].isin(regiao) & df["Categoria"].isin(categoria)]


# --- 4. LAYOUT DA DASHBOARD COM ABAS ---

st.title("🚀 Dashboard Interativa com Gemini AI")
st.markdown("Use as abas abaixo para navegar entre a visualização de dados e a análise com Inteligência Artificial.")

tab1, tab2, tab3 = st.tabs(["📊 Resumo Visual", "🤖 Análise com IA", "💬 Pergunte aos Dados"])

# --- ABA 1: RESUMO VISUAL ---
with tab1:
    st.header("Principais Indicadores (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vendas Totais", f"R$ {df_filtrado['Vendas'].sum():,.2f}")
    col2.metric("Ticket Médio", f"R$ {df_filtrado['Vendas'].mean():,.2f}")
    col3.metric("Nº de Vendas", df_filtrado.shape[0])

    st.markdown("---")
    st.header("Visualizações Gráficas")
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.subheader("Vendas por Categoria")