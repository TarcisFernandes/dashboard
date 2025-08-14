# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os

# --- 1. CONFIGURAÇÃO INICIAL E DE SEGURANÇA ---

# =================================================================================
# ATENÇÃO! ALERTA DE SEGURANÇA!
# A sua chave API está inserida diretamente no código abaixo.
# Isto é APENAS PARA TESTES LOCAIS.
# NÃO COMPARTILHE ESTE ARQUIVO COM NINGUÉM E NÃO O ENVIE PARA O GITHUB.
# Qualquer pessoa com esta chave pode usá-la em seu nome.
# =================================================================================
SUA_CHAVE_API = "AIzaSyAXWKA9oEpsPu9UbzCFaiiThiAKOuLLN7o" 

# Configura a página do Streamlit
st.set_page_config(
    page_title="Dashboard de Vendas com IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configura a API do Gemini com a chave inserida diretamente
if SUA_CHAVE_API and SUA_CHAVE_API != "SUA_CHAVE_API_AQUI":
    try:
        genai.configure(api_key=SUA_CHAVE_API)
        GEMINI_CONFIGURADO = True
    except Exception as e:
        st.error(f"Erro ao configurar a API do Gemini: {e}")
        GEMINI_CONFIGURADO = False
else:
    st.error("ERRO: A chave da API não foi inserida no código. Edite o arquivo app.py e substitua 'SUA_CHAVE_API_AQUI' pela sua chave real.")
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
        fig_cat = px.bar(df_filtrado.groupby('Categoria')['Vendas'].sum().reset_index(), x='Categoria', y='Vendas')
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_graf2:
        st.subheader("Vendas por Região")
        fig_reg = px.pie(df_filtrado.groupby('Regiao')['Vendas'].sum().reset_index(), names='Regiao', values='Vendas')
        st.plotly_chart(fig_reg, use_container_width=True)

# --- ABA 2: ANÁLISE COM IA ---
with tab2:
    st.header("Análise Qualitativa dos Dados com Gemini")
    st.markdown("Clique no botão abaixo para que o Gemini gere um resumo executivo sobre os dados filtrados.")

    if st.button("Gerar Resumo Analítico", disabled=not GEMINI_CONFIGURADO):
        with st.spinner("Gemini está pensando... 🧠"):
            # Prompt para o Gemini: instrui a IA a atuar como um analista de dados
            prompt = f"""
            Você é um analista de dados sênior. Analise o resumo dos dados de vendas a seguir:
            Resumo estatístico:
            {df_filtrado.describe().to_string()}
            
            Vendas por categoria:
            {df_filtrado.groupby('Categoria')['Vendas'].sum().to_string()}

            Com base nesses números, escreva uma análise curta (3 parágrafos) com os principais insights,
            oportunidades e pontos de atenção para uma reunião de diretoria. Use um tom profissional e direto.
            """
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.markdown("---")
            st.subheader("Análise do Gemini:")
            st.markdown(response.text)
    elif not GEMINI_CONFIGURADO:
        st.warning("A funcionalidade de IA está desabilitada. Verifique a configuração da sua chave API no topo do arquivo app.py.")

# --- ABA 3: PERGUNTE AOS DADOS ---
with tab3:
    st.header("Converse com seus Dados")
    st.markdown("Faça uma pergunta em linguagem natural sobre os dados. Exemplos: 'Qual produto vendeu mais?' ou 'qual a média de vendas da região sul?'")

    pergunta_usuario = st.text_input("Sua pergunta:", key="pergunta_ia")

    if pergunta_usuario and GEMINI_CONFIGURADO:
        with st.spinner("Gemini está consultando os dados para você... 🕵️"):
            # Prompt avançado: instrui a IA a traduzir a pergunta para código Python
            prompt = f"""
            Você é um expert em Pandas. O usuário está trabalhando com um DataFrame chamado 'df_filtrado'
            com as seguintes colunas: {df_filtrado.columns.to_list()}.
            
            Converta a seguinte pergunta do usuário em um único comando de código Python que possa ser
            executado para encontrar a resposta. Retorne APENAS o código, sem explicações, sem `print()`,
            sem aspas ou formatação de código.
            
            Pergunta: "{pergunta_usuario}"
            Código Python:
            """
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            codigo_gerado = response.text.strip()
            
            st.markdown("---")
            st.write("🔍 **Gemini interpretou sua pergunta e gerou o seguinte código para encontrar a resposta:**")
            st.code(codigo_gerado, language='python')

            try:
                # ATENÇÃO: Executar código gerado dinamicamente é um risco.
                # Para esta ferramenta de apresentação interna, é aceitável, mas requer cuidado.
                resultado = eval(codigo_gerado, {"df_filtrado": df_filtrado, "pd": pd})
                st.write("✅ **Resultado:**")
                st.write(resultado)
            except Exception as e:
                st.error(f"Não foi possível executar a consulta. O Gemini pode ter gerado um código inválido. Tente reformular a pergunta. Erro: {e}")

    elif pergunta_usuario and not GEMINI_CONFIGURADO:
        st.warning("A funcionalidade de IA está desabilitada. Verifique a configuração da sua chave API no topo do arquivo app.py.")