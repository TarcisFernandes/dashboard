# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import camelot
import os
import re

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

st.set_page_config(
    page_title="Análise de Auditoria de Vendas",
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
def limpar_e_estruturar_dados(df_bruto):
    """Função especializada para limpar e reestruturar o DataFrame extraído do PDF de auditoria."""
    # Renomeia as colunas extraídas, resolvendo o problema de 'ID' duplicado
    df_bruto.columns = ['ID_Cliente', 'Razao_Social_Completa', 'Vendedor', 'Valor_Comissao', 'ID_Contrato', 'Status_Contrato', 'Data_Cancelamento']

    # Itera sobre cada linha para corrigir os dados misturados
    for index, row in df_bruto.iterrows():
        # Tenta encontrar um valor de comissão (ex: 15.00) no campo 'Razao_Social_Completa'
        match = re.search(r'(\d+\.\d{2})$', str(row['Razao_Social_Completa']).strip())
        if match:
            comissao_extraida = float(match.group(1))
            # Atualiza o valor da comissão na sua coluna correta
            df_bruto.loc[index, 'Valor_Comissao'] = comissao_extraida
            
            # Remove o valor da comissão e o que estiver depois dele do campo principal
            resto_string = str(row['Razao_Social_Completa']).strip()[:match.start()].strip()
            
            # A heurística aqui é complexa. Vamos assumir que nomes de vendedores conhecidos podem ser encontrados.
            # Por simplicidade, vamos apenas limpar o campo por enquanto.
            # Uma lógica mais avançada poderia ser adicionada aqui para separar cliente de vendedor.
            df_bruto.loc[index, 'Razao_Social_Completa'] = resto_string

    # Converte os tipos de dados para os corretos
    df_bruto['Valor_Comissao'] = pd.to_numeric(df_bruto['Valor_Comissao'], errors='coerce')
    df_bruto['Data_Cancelamento'] = pd.to_datetime(df_bruto['Data_Cancelamento'], format='%d/%m/%Y', errors='coerce')
    df_bruto.rename(columns={'Razao_Social_Completa': 'Razao_Social'}, inplace=True)
    
    # Remove linhas onde a comissão não pôde ser convertida para número
    df_bruto.dropna(subset=['Valor_Comissao'], inplace=True)
    
    return df_bruto


# --- SIDEBAR E LÓGICA DE UPLOAD ---

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader(
    "Faça o upload do seu relatório de auditoria (PDF)",
    type=['pdf']
)

df = None

if uploaded_file is not None:
    with st.spinner("Lendo e processando o relatório PDF... Isso pode levar alguns minutos."):
        try:
            temp_pdf_path = f"./temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Usar 'lattice' pode ser melhor para tabelas com linhas visíveis, 'stream' para as baseadas em espaço
            tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='stream')
            os.remove(temp_pdf_path)
            
            if tables:
                df_bruto = pd.concat([table.df for table in tables], ignore_index=True)
                
                # Remove o cabeçalho repetido em cada página
                df_bruto = df_bruto[~df_bruto[0].isin(['ID', ''])]
                
                df = limpar_e_estruturar_dados(df_bruto)
                st.sidebar.success(f"Relatório processado! {len(df)} vendas analisadas.")
            else:
                st.error("Nenhuma tabela foi encontrada neste arquivo PDF.")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o PDF: {e}")

# --- DASHBOARD PRINCIPAL ---

if df is not None:
    st.title("📊 Dashboard de Auditoria de Vendas e Comissões")
    st.markdown(f"Análise do Relatório: **{uploaded_file.name}**")

    # --- FILTROS ---
    st.sidebar.markdown("---")
    st.sidebar.title("Filtros")
    
    vendedores = df['Vendedor'].dropna().unique()
    status = df['Status_Contrato'].dropna().unique()

    vendedor_selecionado = st.sidebar.multiselect("Filtrar por Vendedor:", options=vendedores, default=vendedores)
    status_selecionado = st.sidebar.multiselect("Filtrar por Status do Contrato:", options=status, default=status)

    df_filtrado = df[df['Vendedor'].isin(vendedor_selecionado) & df['Status_Contrato'].isin(status_selecionado)]

    # --- ABAS DA DASHBOARD ---
    tab1, tab2, tab3 = st.tabs(["📈 Resumo Geral", "🧑‍💼 Análise por Vendedor", "🤖 Análise com IA"])

    with tab1:
        st.header("Visão Geral das Comissões")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Comissão Total", f"R$ {df_filtrado['Valor_Comissao'].sum():,.2f}")
        col2.metric("Nº de Vendas", f"{len(df_filtrado)}")
        col3.metric("Comissão Média", f"R$ {df_filtrado['Valor_Comissao'].mean():,.2f}")
        
        contratos_inativos = df_filtrado[df_filtrado['Status_Contrato'] == 'Inativo'].shape[0]
        col4.metric("Contratos Cancelados", f"{contratos_inativos}")

        st.markdown("---")
        
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.subheader("Distribuição de Status dos Contratos")
            status_counts = df_filtrado['Status_Contrato'].value_counts()
            fig_pie = px.pie(status_counts, values=status_counts.values, names=status_counts.index, hole=.3)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_graf2:
            st.subheader("Cancelamentos ao Longo do Tempo")
            cancelamentos = df_filtrado[df_filtrado['Data_Cancelamento'].notna()]
            cancel_por_mes = cancelamentos.set_index('Data_Cancelamento').resample('M').size().rename("Nº de Cancelamentos")
            if not cancel_por_mes.empty:
                st.line_chart(cancel_por_mes)
            else:
                st.info("Não há dados de cancelamento para o período filtrado.")

    with tab2:
        st.header("Performance por Vendedor")
        
        # Calcula as métricas por vendedor
        vendas_por_vendedor = df_filtrado.groupby('Vendedor').agg(
            Comissao_Total=('Valor_Comissao', 'sum'),
            Numero_de_Vendas=('Vendedor', 'count')
        ).reset_index().sort_values('Comissao_Total', ascending=False)

        st.subheader("Top Vendedores por Comissão Total")
        fig_bar_comissao = px.bar(vendas_por_vendedor.head(10), x='Vendedor', y='Comissao_Total', text_auto='.2s')
        fig_bar_comissao.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig_bar_comissao, use_container_width=True)

        st.subheader("Top Vendedores por Número de Vendas")
        fig_bar_vendas = px.bar(vendas_por_vendedor.sort_values('Numero_de_Vendas', ascending=False).head(10), x='Vendedor', y='Numero_de_Vendas')
        st.plotly_chart(fig_bar_vendas, use_container_width=True)

    with tab3:
        st.header("Análise Avançada com IA")
        st.info("A IA do Gemini pode analisar os dados filtrados para encontrar insights.")

        if st.button("Gerar Análise com IA", disabled=not GEMINI_CONFIGURADO):
            with st.spinner("Gemini está pensando... 🧠"):
                
                # Prepara um resumo dos dados para a IA
                resumo_vendedores = df_filtrado.groupby('Vendedor')['Valor_Comissao'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False).to_string()

                prompt = f"""
                Você é um diretor de vendas analisando um relatório de comissões da empresa SUPERTEC TELECOM.
                Os dados filtrados mostram o seguinte resumo de performance por vendedor:
                {resumo_vendedores}

                Com base nesses números, escreva uma análise executiva curta (3 a 4 parágrafos) destacando:
                1. Os vendedores com melhor performance (em valor total e em quantidade de vendas).
                2. Quaisquer discrepâncias interessantes (ex: vendedor com muitas vendas mas baixa comissão total, ou vice-versa).
                3. Uma recomendação de ação baseada nos dados.
                """
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                st.subheader("Análise Gerada pelo Gemini:")
                st.markdown(response.text)

else:
    st.info("👋 Bem-vindo à Dashboard de Auditoria! Por favor, faça o upload do seu relatório PDF para começar.")