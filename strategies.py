import pandas as pd
import numpy as np

def calc_rsi(series, period=2):
    """Calcula o RSI (Índice de Força Relativa) usando a média de Wilder."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    """Calcula o ATR (Average True Range) para dimensionamento de risco e stops."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def precompute_indicators(stocks_dict, ibov_df):
    """
    Pré-calcula todos os indicadores técnicos necessários para as ações e o IBOVESPA.
    """
    print("Pré-calculando indicadores técnicos...")
    
    # 1. Indicadores do IBOVESPA (Regime)
    ibov = ibov_df.copy()
    ibov['EMA21'] = ibov['Close'].ewm(span=21, adjust=False).mean()
    ibov['SMA200'] = ibov['Close'].rolling(200).mean()
    
    # 2. Processar cada ação
    processed_stocks = {}
    for ticker, df in stocks_dict.items():
        df = df.copy()
        
        # Baselines e Indicadores Gerais
        df['ATR'] = calc_atr(df, 14)
        df['SMA200'] = df['Close'].rolling(200).mean()
        
        # Cruzamento de Média (Baseline 1)
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # Bandas de Bollinger (Baseline 2)
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['StdDev20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + 2 * df['StdDev20']
        df['BB_Lower'] = df['SMA20'] - 2 * df['StdDev20']
        df['RSI14'] = calc_rsi(df['Close'], 14)
        
        # RSI(2) Larry Connors (Baseline 3)
        df['RSI2'] = calc_rsi(df['Close'], 2)
        df['Max2'] = df['High'].rolling(2).max()
        
        # Canais Donchian (Baseline 4)
        df['Donchian_High20'] = df['High'].shift(1).rolling(20).max()
        df['Donchian_Low10'] = df['Low'].shift(1).rolling(10).min()
        
        # B3-RAVTH (Evoluída) - Indicadores específicos
        df['AvgVolume5'] = df['Volume'].rolling(5).mean()
        
        # Força Relativa de 21 dias comparada ao IBOV
        ibov_ret = ibov['Close'].pct_change(21)
        stock_ret = df['Close'].pct_change(21)
        df = df.join(ibov_ret.to_frame('IBOV_Ret21'), how='left')
        df['RS'] = stock_ret - df['IBOV_Ret21']
        
        # Juntar dados de regime do IBOV
        ibov_regime = ibov[['Close', 'EMA21', 'SMA200']].rename(
            columns={'Close': 'IBOV_Close', 'EMA21': 'IBOV_EMA21', 'SMA200': 'IBOV_SMA200'}
        )
        df = df.join(ibov_regime, how='left')
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        processed_stocks[ticker] = df
        
    return processed_stocks, ibov

def get_signals_for_day(strategy_name, ticker, df, idx_day):
    """
    Retorna se há sinal de compra (LONG) ou venda (EXIT) no dia 'idx_day' para um ativo específico.
    Retorna: (is_buy, is_sell, stop_loss_price, ranking_score)
    """
    if idx_day < 1 or idx_day >= len(df):
        return False, False, 0.0, 0.0
        
    row = df.iloc[idx_day]
    prev_row = df.iloc[idx_day - 1]
    
    is_buy = False
    is_sell = False
    stop_loss = 0.0
    ranking_score = 0.0
    
    # -------------------------------------------------------------------------
    # ESTRATÉGIA 1: EMA CROSSOVER (Média 9 e 21)
    # -------------------------------------------------------------------------
    if strategy_name == 'EMA_Crossover':
        if prev_row['EMA9'] <= prev_row['EMA21'] and row['EMA9'] > row['EMA21']:
            is_buy = True
            stop_loss = row['Close'] - 3.5 * row['ATR']
            ranking_score = row['EMA9'] / row['EMA21']
        elif row['EMA9'] < row['EMA21']:
            is_sell = True
            
    # -------------------------------------------------------------------------
    # ESTRATÉGIA 2: BOLLINGER BANDS (Reversão à Média)
    # -------------------------------------------------------------------------
    elif strategy_name == 'Bollinger_Bands':
        if row['Close'] < row['BB_Lower'] and row['RSI14'] < 30:
            is_buy = True
            stop_loss = row['Close'] - 3.5 * row['ATR']
            ranking_score = 100 - row['RSI14']
        elif row['Close'] > row['SMA20']:
            is_sell = True
            
    # -------------------------------------------------------------------------
    # ESTRATÉGIA 3: RSI(2) LARRY CONNORS (Reversão em Tendência)
    # -------------------------------------------------------------------------
    elif strategy_name == 'Rsi2_Connors':
        if row['Close'] > row['SMA200'] and row['RSI2'] < 10:
            is_buy = True
            stop_loss = row['Close'] - 4.0 * row['ATR']
            ranking_score = 10 - row['RSI2']
        elif row['Close'] > prev_row['Max2']:
            is_sell = True
            
    # -------------------------------------------------------------------------
    # ESTRATÉGIA 4: DONCHIAN BREAKOUT (Rompimento)
    # -------------------------------------------------------------------------
    elif strategy_name == 'Donchian_Breakout':
        if row['Close'] > row['Donchian_High20']:
            is_buy = True
            stop_loss = row['Close'] - 3.0 * row['ATR']
            ranking_score = row['Close'] / row['Donchian_High20']
        elif row['Close'] < row['Donchian_Low10']:
            is_sell = True
            
    # -------------------------------------------------------------------------
    # ESTRATÉGIA 5: B3-RAVTH v2 (Evoluída e Otimizada)
    # -------------------------------------------------------------------------
    elif strategy_name == 'B3_RAVTH':
        ibov_close = row['IBOV_Close']
        ibov_sma200 = row['IBOV_SMA200']
        ibov_ema21 = row['IBOV_EMA21']
        
        is_bull_trend = ibov_close > ibov_sma200
        
        # A) Bear Market de Índice: Saída imediata de posições compradas para proteção
        if not is_bull_trend:
            is_sell = True
            is_buy = False
            
        # B) Bull/Sideways (IBOV acima da SMA200)
        else:
            is_strong = ibov_close > ibov_ema21
            
            # 1. Regime de Forte Alta: Líderes em Pullback Profundo
            if is_strong:
                if row['Close'] > row['SMA200'] and row['RS'] > 0.05 and row['RSI2'] < 10:
                    is_buy = True
                    stop_loss = row['Close'] - 3.5 * row['ATR']
                    ranking_score = row['RS'] * 100
                    
            # 2. Regime Lateral/Consolidação: Reversão à Média com Exaustão
            else:
                if row['Close'] > row['SMA200'] and row['RSI2'] < 5 and row['Volume'] < row['AvgVolume5']:
                    is_buy = True
                    stop_loss = row['Close'] - 4.0 * row['ATR']
                    ranking_score = 10 - row['RSI2']
                    
            # Saídas normais do B3-RAVTH v2
            if row['Close'] > prev_row['Max2'] or row['RSI2'] > 75:
                is_sell = True
                
    return is_buy, is_sell, stop_loss, ranking_score
