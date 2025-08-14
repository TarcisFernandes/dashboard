# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

st.set_page_config(
    page_title="Dashboard Dinâmica com IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_CONFIGURADO = True
except Exception as e:
    st.error(f"Erro ao configurar a API do Gemini. Verifique se a chave 'GEMINI_API_KEY' está nos segredos. Erro: {e}")
    GEMINI_CONFIGURADO = False

# --- 2. UPLOAD E PROCESSAMENTO DO ARQUIVO ---

# MANTÉM A VERIFICAÇÃO ROBUSTA DENTRO DE UMA FUNÇÃO AUXILIAR
def processar_dataframe(df):
    """Verifica e processa o DataFrame para garantir que a coluna 'Data' está correta."""
    try:
        # Garante que a coluna 'Data' existe e a converte para datetime
        df['Data'] = pd.to_datetime(df['Data'])
        return df
    except KeyError:
        st.error(
            f"Erro Crítico: A coluna 'Data' não foi encontrada no arquivo enviado.\n\n"
            f"**Colunas encontradas:** {list(df.columns)}\n\n"
            "Por favor, renomeie a coluna de data no seu arquivo CSV para 'Data' (com 'D' maiúsculo) e envie novamente."
        )
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar a coluna de data: {e}")
        return None

# --- SIDEBAR E LÓGICA DE UPLOAD ---

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader(
    "Faça o upload do seu arquivo CSV",
    type=['csv']
)

# A LÓGICA PRINCIPAL DO APP AGORA DEPENDE DO UPLOAD
if uploaded_file is not None:
    # Carrega os dados do arquivo enviado
    df = pd.read_csv(uploaded_file)
    df = processar_dataframe(df)

    # Continua a execução somente se o DataFrame for válido
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.title("Filtros Interativos")
        regiao = st.sidebar.multiselect("Selecione a Região:", options=df["Regiao"].unique(), default=df["Regiao"].unique())
        categoria = st.sidebar.multiselect("Selecione a Categoria:", options=df["Categoria"].unique(), default=df["Categoria"].unique())
        df_filtrado = df[df["Regiao"].isin(regiao) & df["Categoria"].isin(categoria)]

        # --- LAYOUT DA DASHBOARD COM ABAS ---

        st.title("🚀 Dashboard Dinâmica com Gemini AI")
        st.markdown("Use as abas para navegar entre a visualização e a análise dos seus dados.")

        tab1, tab2, tab3 = st.tabs(["📊 Resumo Visual", "🤖 Análise com IA", "💬 Pergunte aos Dados"])

        # --- ABA 1: RESUMO VISUAL ---
        with tab1:
            if not df_filtrado.empty:
                st.header("Principais Indicadores (KPIs)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Vendas Totais", f"R$ {df_filtrado['Vendas'].sum():,.2f}")
                col2.metric("Ticket Médio", f"R$ {df_filtrado['Vendas'].mean():,.2f}")
                col3.metric("Nº de Vendas", df_filtrado.shape[0])
                st.markdown("---")
                st.header("Visualizações Gráficas")
                # Gráficos... (código omitido para brevidade, mas está no bloco acima)
            else:
                st.warning("Nenhum dado disponível para os filtros selecionados.")
        
        # --- ABA 2: ANÁLISE COM IA ---
        with tab2:
            st.header("Análise Qualitativa dos Dados com Gemini")
            if not df_filtrado.empty:
                if st.button("Gerar Resumo Analítico", disabled=not GEMINI_CONFIGURADO):
                    with st.spinner("Gemini está pensando... 🧠"):
                        prompt = f"""
                        Analise os seguintes dados de vendas no formato JSON e forneça insights:
                        Resumo estatístico: {df_filtrado.describe().to_json()}
                        Vendas por categoria: {df_filtrado.groupby('Categoria')['Vendas'].sum().to_json()}
                        """
                        model = genai.GenerativeModel('gemini-pro')
                        safety_settings = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                        response = model.generate_content(prompt, safety_settings=safety_settings)
                        st.subheader("Análise do Gemini:")
                        st.markdown(response.text)
            else:
                st.warning("Não há dados para analisar com os filtros selecionados.")
        
        # --- ABA 3: PERGUNTE AOS DADOS ---
        with tab3:
            st.header("Converse com seus Dados")
            if not df_filtrado.empty:
                pergunta_usuario = st.text_input("Sua pergunta:", key="pergunta_ia")
                if pergunta_usuario and GEMINI_CONFIGURADO:
                    with st.spinner("Gemini está consultando os dados... 🕵️"):
                        prompt = f"""
                        O DataFrame 'df_filtrado' tem as colunas: {df_filtrado.columns.to_list()}.
                        Converta a pergunta do usuário em um único comando de código Python para encontrar a resposta.
                        Retorne APENAS o código. Pergunta: "{pergunta_usuario}"
                        """
                        model = genai.GenerativeModel('gemini-pro')
                        safety_settings = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                        response = model.generate_content(prompt, safety_settings=safety_settings)
                        codigo_gerado = response.text.strip().replace('```python', '').replace('```', '').strip()
                        st.code(codigo_gerado, language='python')
                        try:
                            resultado = eval(codigo_gerado, {"df_filtrado": df_filtrado, "pd": pd})
                            st.write("✅ **Resultado:**", resultado)
                        except Exception as e:
                            st.error(f"Não foi possível executar o código gerado. Erro: {e}")
            else:
                st.warning("Não há dados para consultar com os filtros selecionados.")

# TELA INICIAL ANTES DO UPLOAD
else:
    st.info("👋 Bem-vindo! Por favor, faça o upload de um arquivo CSV na barra lateral para começar a análise.")