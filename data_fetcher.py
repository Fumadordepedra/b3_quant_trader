import yfinance as yf
import pandas as pd
import numpy as np
import os
import urllib.request
import re

# Lista estática de fallback de 45 principais ativos da B3
STATIC_TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 'B3SA3.SA', 
    'PRIO3.SA', 'WEGE3.SA', 'ABEV3.SA', 'ITSA4.SA', 'ELET3.SA', 'EQTL3.SA', 
    'SUZB3.SA', 'GGBR4.SA', 'VIVT3.SA', 'SBSP3.SA', 'CMIG4.SA', 'CPLE6.SA', 
    'CSAN3.SA', 'JBSS3.SA', 'LREN3.SA', 'RENT3.SA', 'RADL3.SA', 'TOTS3.SA', 
    'HAPV3.SA', 'MULT3.SA', 'CYRE3.SA', 'RAIL3.SA', 'USIM5.SA', 'MRVE3.SA', 
    'BEEF3.SA', 'GOAU4.SA', 'ASAI3.SA', 'CRFB3.SA', 'NTCO3.SA', 'EGIE3.SA', 
    'TAEE11.SA', 'TRPL4.SA', 'CPFE3.SA', 'FLRY3.SA', 'SLCE3.SA', 'UGPA3.SA', 
    'TIMS3.SA', 'BOVA11.SA', 'SMAL11.SA'
]

# Inicialmente, o TICKERS global aponta para o fallback
TICKERS = STATIC_TICKERS.copy()

IBOV_TICKER = '^BVSP'
START_DATE = '2018-01-01'
END_DATE = '2026-05-20'
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(DATA_DIR, 'b3_historical_data.pkl')

def fetch_all_liquid_b3_tickers(min_volume=30_000_000, min_price=1.00):
    """
    Varre a internet (via Fundamentus) para descobrir dinamicamente todas as ações
    negociadas na B3. Filtra automaticamente por liquidez e preço (exclui penny stocks).
    Retorna uma lista de strings compatíveis com yfinance (ex: 'PETR4.SA').
    """
    print("\n" + "="*70)
    print(" INICIANDO DESCOBERTA AUTOMÁTICA DE ATIVOS NA B3 (VIA WEB) ")
    print("="*70)
    
    url = 'http://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('latin-1')
            
        # Extrair linhas da tabela
        rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL)
        discovered_tickers = []
        
        for row in rows:
            # Procura a tag do papel
            ticker_match = re.search(r'detalhes\.php\?papel=([A-Z0-9]{4,6})', row)
            if not ticker_match:
                continue
            ticker = ticker_match.group(1)
            
            # Extrair todas as colunas
            cols = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
            if len(cols) >= 19:
                try:
                    # Coluna 1 (índice 1): Cotação
                    price = float(cols[1].strip().replace('.', '').replace(',', '.'))
                    # Coluna 18 (índice 18): Volume Financeiro Médio de 2m
                    volume = float(cols[18].strip().replace('.', '').replace(',', '.'))
                    
                    # Filtros de Segurança Quantitativos:
                    # 1. Excluir Penny Stocks (ações < R$ 1,00) - evitamos quebras e alta especulação
                    # 2. Excluir Ativos sem Liquidez Mínima de Volume Diário Médio (default R$ 30 milhões)
                    if price >= min_price and volume >= min_volume:
                        discovered_tickers.append((ticker, volume, price))
                except ValueError:
                    continue
                    
        # Ordenar os ativos pela liquidez (volume) de forma decrescente
        discovered_tickers = sorted(discovered_tickers, key=lambda x: x[1], reverse=True)
        
        # Converter para formato yfinance (.SA)
        final_list = [f"{item[0]}.SA" for item in discovered_tickers]
        
        # Adicionar ETFs que não aparecem na triagem de ações normais mas são essenciais
        for etf in ['BOVA11.SA', 'SMAL11.SA']:
            if etf not in final_list:
                final_list.append(etf)
                
        print(f"[OK] Descoberta automática concluída! Encontrados {len(final_list)} ativos líquidos e operáveis na B3.")
        print(f"Top 5 ativos mais líquidos detectados: {', '.join([x.replace('.SA','') for x in final_list[:5]])}")
        return final_list
        
    except Exception as e:
        print(f"[Falha na Descoberta Web] Erro ao conectar ou processar Fundamentus: {str(e)}")
        print("[*] Utilizando a lista estática curada de fallback de 45 principais ativos.")
        return STATIC_TICKERS.copy()

def download_and_filter_data(min_avg_volume=30_000_000):
    """
    Realiza o download dos dados da B3 usando a descoberta automática ou fallback,
    e salva o cache físico local para otimizar os tempos de execução.
    """
    global TICKERS
    
    # 1. Buscar a lista de ativos dinamicamente da internet!
    TICKERS = fetch_all_liquid_b3_tickers(min_volume=min_avg_volume)
    
    print("\n" + "="*60)
    print(" ATUALIZANDO DADOS HISTÓRICOS B3 ")
    print("="*60)
    
    # Baixar o IBOVESPA (Benchmark e Regime)
    print(f"Baixando dados do IBOVESPA ({IBOV_TICKER}) de {START_DATE} até {END_DATE}...")
    ibov = yf.download(IBOV_TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov.columns = ibov.columns.get_level_values(0)
    
    ibov = ibov[['Close', 'High', 'Low', 'Open', 'Volume']].dropna()
    ibov.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # Baixar os ativos descobertos
    print(f"Baixando dados para {len(TICKERS)} ativos da B3...")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)
    
    valid_data = {}
    total_downloaded = 0
    
    # Processar cada ticker individualmente
    for ticker in TICKERS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs(ticker, level=1, axis=1).copy()
            else:
                if len(TICKERS) == 1:
                    df = data.copy()
                else:
                    continue
            
            df.dropna(subset=['Close'], inplace=True)
            
            # Requisito de dados mínimos (pelo menos 5 anos ~ 1200 pregões para consistência do backtest)
            if len(df) < 1100:
                continue
                
            total_downloaded += 1
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            # Adicionar VolFin para compatibilidade com o analisador
            df['VolFin'] = df['Close'] * df['Volume']
            valid_data[ticker] = df
            
        except Exception as e:
            pass
            
    print(f"\n[OK] Coleta concluída! {total_downloaded} ativos foram indexados no banco de dados local.")
    
    # Salvar cache
    cache_pack = {
        'ibov': ibov,
        'stocks': valid_data
    }
    
    pd.to_pickle(cache_pack, CACHE_FILE)
    print(f"Dados salvos com sucesso em: {CACHE_FILE}")
    return cache_pack

if __name__ == "__main__":
    download_and_filter_data()
