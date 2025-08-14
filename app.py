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
                
                # Lê todas as páginas do PDF
                tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='stream')
                os.remove(temp_pdf_path)
                
                if len(tables) > 0:
                    st.sidebar.success(f"{len(tables)} segmento(s) de tabela encontrado(s) no PDF.")
                    
                    # --- NOVO: LÓGICA DE CONCATENAÇÃO DE TABELAS ---
                    # Assume que todas as tabelas encontradas são partes de uma única tabela maior.
                    
                    # Primeiro, limpa cada pedaço de tabela (promove o cabeçalho)
                    cleaned_dfs = []
                    for table in tables:
                        temp_df = table.df
                        # Verifica se a tabela não está vazia
                        if not temp_df.empty:
                            # Usa a primeira linha como cabeçalho
                            new_header = temp_df.iloc[0] 
                            temp_df = temp_df[1:] 
                            temp_df.columns = new_header
                            cleaned_dfs.append(temp_df)
                    
                    # Concatena todos os DataFrames limpos em um só
                    if cleaned_dfs:
                        df = pd.concat(cleaned_dfs, ignore_index=True)
                        st.sidebar.info(f"As tabelas foram combinadas, resultando em {len(df)} linhas.")
                    else:
                        st.error("As tabelas encontradas no PDF estavam vazias.")

                else:
                    st.error("Nenhuma tabela foi encontrada neste arquivo PDF.")

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o PDF: {e}")

# --- A PARTIR DAQUI, O CÓDIGO PROCESSA O DATAFRAME `df` ---
if df is not None:
    # Renomeia colunas para serem strings e remove espaços em branco
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
        if col in date_cols:
            # Garante que a comparação seja feita corretamente com datas
            if pd.api.types.is_datetime64_any_dtype(df_filtrado[col]):
                 df_filtrado = df_filtrado[(df_filtrado[col].dt.date >= valores[0]) & (df_filtrado[col].dt.date <= valores[1])]
        else:
            df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

    st.title("🚀 Dashboard Analítica Gerada por IA")
    st.markdown(f"Análise do arquivo: `{uploaded_file.name}`")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Dinâmica", "🤖 Análise com IA", "🔍 Visualizar Dados"])

    # (O restante do código das abas continua o mesmo)
    with tab1:
        # Lógica da Aba 1
        pass
    with tab2:
        # Lógica da Aba 2
        pass
    with tab3:
        # Lógica da Aba 3
        st.header("Visualização dos Dados Filtrados")
        st.dataframe(df_filtrado)


else:
    st.info("👋 Bem-vindo! Por favor, faça o upload de um arquivo CSV ou PDF para começar.")