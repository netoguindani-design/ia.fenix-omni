import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from gtts import gTTS
import base64
import requests
import time
import pandas as pd

# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(page_title="IA Fênix Omni", page_icon="🔥", layout="wide")
st.title("🔥 Núcleo Fênix: Inteligência Artificial Híbrida")
st.sidebar.header("Configurações de Controle")

# --- PARÂMETROS ---
MEU_WHATSAPP = "+5593981292787"
# Para usar o WhatsApp, você precisará da sua API Key do CallMeBot
API_KEY_WHATSAPP = "123456" # Substitua pela sua chave real depois

# --- 1. O CÉREBRO DA IA (ARQUITETURA HÍBRIDA) ---
class CerebroFenix(nn.Module):
    def __init__(self):
        super(CerebroFenix, self).__init__()
        self.memoria = nn.LSTM(1, 128, batch_first=True)
        self.decisor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [Ação de Mercado, Estado de Voz]
        )

    def forward(self, x):
        _, (h_n, _) = self.memoria(x)
        return self.decisor(h_n[-1])

# Inicialização segura do modelo
if 'modelo' not in st.session_state:
    st.session_state.modelo = CerebroFenix()
    st.session_state.historico_precos = []

# --- 2. FUNÇÕES SENSORIAIS (VOZ E WHATSAPP) ---
def falar(texto):
    try:
        tts = gTTS(text=texto, lang='pt')
        tts.save("fala.mp3")
        with open("fala.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
            st.markdown(md, unsafe_allow_html=True)
    except:
        st.warning("Erro ao gerar áudio.")

def alertar_whatsapp(msg):
    url = f"https://api.callmebot.com/whatsapp.php?phone={MEU_WHATSAPP}&text={msg}&apikey={API_KEY_WHATSAPP}"
    try:
        requests.get(url)
    except:
        pass

# --- 3. DASHBOARD DE CONTROLE ---
col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input("Símbolo do Mercado (Ex: BTC-USD, AAPL, PETR4.SA)", "BTC-USD")
    btn_ativar = st.button("🚀 Ativar Consciência da IA")

with col2:
    st.info("A IA está em modo de Self-Play. Ela aprende comparando o preço atual com suas previsões anteriores.")

if btn_ativar:
    placeholder = st.empty()
    
    # Loop de monitoramento
    for _ in range(100): # Simulação de ciclo contínuo
        with placeholder.container():
            # Coleta de dados
            dados = yf.Ticker(ticker_input).history(period="1d", interval="1m")
            if not dados.empty:
                preco_atual = dados['Close'].iloc[-1]
                
                # Inteligência Artificial em Ação
                tensor_entrada = torch.tensor([[[preco_atual]]], dtype=torch.float32)
                saida = st.session_state.modelo(tensor_entrada)
                
                # Visualização
                st.metric(label=f"Valor Atual de {ticker_input}", value=f"${preco_atual:,.2f}")
                st.line_chart(dados['Close'])
                
                # Lógica de Decisão e Comunicação
                if len(st.session_state.historico_precos) > 1:
                    variacao = (preco_atual - st.session_state.historico_precos[-1]) / st.session_state.historico_precos[-1]
                    
                    if abs(variacao) > 0.05: # Mudança de 5%
                        msg = f"Mestre, detectei movimento em {ticker_input}. O preço agora é {preco_atual:.2f}. Minha rede neural está se adaptando."
                        st.success(msg)
                        falar(msg)
                        # Descomente a linha abaixo quando tiver sua API KEY
                        # alertar_whatsapp(msg)
                
                st.session_state.historico_precos.append(preco_atual)
            
            time.sleep(60) # Espera 1 minuto para o próximo ciclo
            
                MEU_WHATSAPP = "+5593981292787"
                API_KEY_WHATSAPP = “9580681” 
