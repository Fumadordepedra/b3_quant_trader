import os
import pandas as pd
import numpy as np
from datetime import datetime

# Configurações de caminhos
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(DATA_DIR, 'b3_historical_data.pkl')

def is_etf(ticker):
    return ticker in ['BOVA11.SA', 'SMAL11.SA']

def calc_rsi(series, period=2):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()

def run_backtest_v2(stocks, ibov, strategy_name):
    # Parâmetros
    SLIPPAGE = 0.001
    EMOLUMENTOS = 0.0003
    CDI_ANUAL = 0.10
    CDI_DIARIO = (1.0 + CDI_ANUAL) ** (1.0 / 252.0) - 1.0
    
    all_dates = sorted(list(ibov.index))
    
    cash = 100000.0
    initial_capital = 100000.0
    positions = {} # ticker -> {entry_date, entry_price, shares, stop_loss, days_held}
    
    trades_log = []
    daily_equity = []
    
    loss_carryforward = 0.0
    monthly_sales_volume_stocks = 0.0
    monthly_trades_closed = []
    
    current_month = all_dates[0].month
    
    for day_idx in range(1, len(all_dates)):
        current_date = all_dates[day_idx]
        prev_date = all_dates[day_idx - 1]
        
        # Rendimento CDI
        cash *= (1 + CDI_DIARIO)
        
        # Fechamento de mês: IR
        if current_date.month != current_month:
            cash = process_taxes(cash, monthly_trades_closed, monthly_sales_volume_stocks, loss_carryforward)
            monthly_sales_volume_stocks = 0.0
            monthly_trades_closed = []
            current_month = current_date.month
            
        # Calcular NAV
        open_positions_val = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks[tkr]
            current_price = df_stock.loc[current_date, 'Close'] if current_date in df_stock.index else pos['entry_price']
            open_positions_val += current_price * pos['shares']
        portfolio_value = cash + open_positions_val
        
        # 1. SAÍDAS (VENDAS)
        tickers_in_pos = list(positions.keys())
        for tkr in tickers_in_pos:
            pos = positions[tkr]
            df_stock = stocks[tkr]
            
            if current_date not in df_stock.index:
                continue
                
            row = df_stock.loc[current_date]
            idx_stock = df_stock.index.get_loc(current_date)
            prev_row = df_stock.iloc[idx_stock - 1]
            
            # Condições de Saída por Estratégia
            is_sell = False
            exit_reason = ""
            
            if strategy_name == 'EMA_Crossover':
                if row['EMA9'] < row['EMA21']:
                    is_sell = True
                    exit_reason = "Sinal Saida"
            elif strategy_name == 'Bollinger_Bands':
                if row['Close'] > row['SMA20']:
                    is_sell = True
                    exit_reason = "Sinal Saida"
            elif strategy_name == 'Rsi2_Connors':
                if row['Close'] > prev_row['Max2']:
                    is_sell = True
                    exit_reason = "Sinal Saida"
            elif strategy_name == 'Donchian_Breakout':
                if row['Close'] < row['Donchian_Low10']:
                    is_sell = True
                    exit_reason = "Sinal Saida"
            elif strategy_name == 'B3_RAVTH':
                # RAVTH v2 Saídas adaptativas
                ibov_row = ibov.loc[current_date]
                is_bull = ibov_row['Close'] > ibov_row['SMA200']
                
                if not is_bull:
                    is_sell = True
                    exit_reason = "Bear Market Indice"
                elif row['Close'] > prev_row['Max2'] or row['RSI2'] > 75:
                    is_sell = True
                    exit_reason = "Sinal Saida"
            
            # Verificação de Stop Loss (Wider Stop: 3.5 * ATR para evitar violino)
            stop_dist = 3.5 * row['ATR']
            hit_stop = row['Low'] < pos['stop_loss']
            
            if hit_stop:
                exit_price = min(pos['stop_loss'], row['Open']) * (1 - SLIPPAGE)
                exit_reason = "Stop Loss"
                is_sell = True
            elif is_sell:
                exit_price = row['Close'] * (1 - SLIPPAGE)
            elif pos['days_held'] >= 8: # Time Stop de 8 dias úteis
                exit_price = row['Close'] * (1 - SLIPPAGE)
                exit_reason = "Time Stop"
                is_sell = True
                
            if is_sell:
                gross_val = exit_price * pos['shares']
                fees = gross_val * EMOLUMENTOS
                net_val = gross_val - fees
                
                cash += net_val
                
                entry_val = pos['entry_price'] * pos['shares']
                entry_fees = entry_val * EMOLUMENTOS
                net_profit = net_val - (entry_val + entry_fees)
                
                trade_record = {
                    'Ticker': tkr,
                    'Entry Date': pos['entry_date'],
                    'Exit Date': current_date,
                    'Days': pos['days_held'],
                    'Net Profit': net_profit,
                    'Return %': (exit_price / pos['entry_price'] - 1.0) * 100,
                    'Is_ETF': is_etf(tkr)
                }
                
                trades_log.append(trade_record)
                monthly_trades_closed.append(trade_record)
                if not is_etf(tkr):
                    monthly_sales_volume_stocks += gross_val
                    
                del positions[tkr]
            else:
                positions[tkr]['days_held'] += 1
                
        # Recalcular NAV
        open_positions_val = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks[tkr]
            current_price = df_stock.loc[current_date, 'Close']
            open_positions_val += current_price * pos['shares']
        portfolio_value = cash + open_positions_val
        
        # 2. ENTRADAS (COMPRAS)
        max_positions = 5
        available_slots = max_positions - len(positions)
        
        if available_slots > 0 and cash > 2000.0:
            candidates = []
            for tkr, df_stock in stocks.items():
                if tkr in positions: continue
                if current_date not in df_stock.index: continue
                
                idx = df_stock.index.get_loc(current_date)
                if idx < 1: continue
                
                row = df_stock.iloc[idx]
                prev_row = df_stock.iloc[idx - 1]
                
                is_buy = False
                ranking_score = 0.0
                stop_loss = 0.0
                
                # A) EMA_Crossover
                if strategy_name == 'EMA_Crossover':
                    if prev_row['EMA9'] <= prev_row['EMA21'] and row['EMA9'] > row['EMA21']:
                        is_buy = True
                        ranking_score = row['EMA9'] / row['EMA21']
                        stop_loss = row['Close'] - 3.5 * row['ATR']
                        
                # B) Bollinger_Bands
                elif strategy_name == 'Bollinger_Bands':
                    if row['Close'] < row['BB_Lower'] and row['RSI14'] < 30:
                        is_buy = True
                        ranking_score = 100 - row['RSI14']
                        stop_loss = row['Close'] - 3.5 * row['ATR']
                        
                # C) Rsi2_Connors
                elif strategy_name == 'Rsi2_Connors':
                    if row['Close'] > row['SMA200'] and row['RSI2'] < 10:
                        is_buy = True
                        ranking_score = 10 - row['RSI2']
                        stop_loss = row['Close'] - 4.0 * row['ATR']
                        
                # D) Donchian_Breakout
                elif strategy_name == 'Donchian_Breakout':
                    if row['Close'] > row['Donchian_High20']:
                        is_buy = True
                        ranking_score = row['Close'] / row['Donchian_High20']
                        stop_loss = row['Close'] - 3.0 * row['ATR']
                        
                # E) B3_RAVTH v2 (Evoluída)
                elif strategy_name == 'B3_RAVTH':
                    ibov_row = ibov.loc[current_date]
                    ibov_close = ibov_row['Close']
                    ibov_sma200 = ibov_row['SMA200']
                    ibov_ema21 = ibov_row['EMA21']
                    
                    is_bull = ibov_close > ibov_sma200
                    
                    if is_bull:
                        is_strong = ibov_close > ibov_ema21
                        # 1. Regime de Forte Alta: Líderes em Pullback
                        if is_strong:
                            if row['Close'] > row['SMA200'] and row['RS'] > 0.05 and row['RSI2'] < 10:
                                is_buy = True
                                ranking_score = row['RS'] * 100
                                stop_loss = row['Close'] - 3.5 * row['ATR']
                        # 2. Regime Lateral/Consolidação: Reversão à Média Restrita
                        else:
                            if row['Close'] > row['SMA200'] and row['RSI2'] < 5 and row['Volume'] < row['AvgVolume5']:
                                is_buy = True
                                ranking_score = 10 - row['RSI2']
                                stop_loss = row['Close'] - 4.0 * row['ATR']
                                
                if is_buy:
                    candidates.append({
                        'Ticker': tkr,
                        'Close': row['Close'],
                        'Stop_Loss': stop_loss,
                        'Score': ranking_score
                    })
                    
            candidates = sorted(candidates, key=lambda x: x['Score'], reverse=True)
            for cand in candidates[:available_slots]:
                entry_price = cand['Close'] * (1 + SLIPPAGE)
                
                # Equal Capital Sizing: 20% do capital total da carteira
                pos_value = portfolio_value * 0.20
                if pos_value > cash:
                    pos_value = cash - 50.0
                    
                shares = int(pos_value / entry_price)
                if shares >= 1:
                    cost = shares * entry_price
                    fees = cost * EMOLUMENTOS
                    
                    cash -= (cost + fees)
                    positions[cand['Ticker']] = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'stop_loss': cand['Stop_Loss'],
                        'days_held': 0
                    }
                    
        # Salvar equity
        open_positions_val = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks[tkr]
            current_price = df_stock.loc[current_date, 'Close']
            open_positions_val += current_price * pos['shares']
        portfolio_value = cash + open_positions_val
        
        daily_equity.append({
            'Date': current_date,
            'Total_Equity': portfolio_value
        })
        
    return pd.DataFrame(trades_log), pd.DataFrame(daily_equity)

def process_taxes(cash, trades, volume, loss_cf):
    if not trades: return cash
    stocks_profit = sum(t['Net Profit'] for t in trades if not t['Is_ETF'])
    etfs_profit = sum(t['Net Profit'] for t in trades if t['Is_ETF'])
    
    stock_exempt = volume <= 20000.0
    taxable_stock = max(0.0, stocks_profit) if not stock_exempt else 0.0
    taxable_etf = max(0.0, etfs_profit)
    
    # Prejuízos acumulados
    loss_cf += abs(sum(t['Net Profit'] for t in trades if t['Net Profit'] < 0))
    
    total_taxable = taxable_stock + taxable_etf
    if total_taxable > 0 and loss_cf > 0:
        if loss_cf >= total_taxable:
            loss_cf -= total_taxable
            total_taxable = 0.0
        else:
            total_taxable -= loss_cf
            loss_cf = 0.0
            
    if total_taxable > 0:
        cash -= total_taxable * 0.15
    return cash

def main():
    print(f"Lendo dados de {CACHE_FILE}...")
    cache = pd.read_pickle(CACHE_FILE)
    stocks = cache['stocks']
    ibov = cache['ibov']
    
    # Calcular indicadores para o IBOV
    ibov = ibov.copy()
    ibov['EMA21'] = ibov['Close'].ewm(span=21, adjust=False).mean()
    ibov['SMA200'] = ibov['Close'].rolling(200).mean()
    
    # Calcular indicadores para stocks
    processed_stocks = {}
    for tkr, df in stocks.items():
        df = df.copy()
        df['ATR'] = calc_atr(df, 14)
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_Lower'] = df['SMA20'] - 2 * df['Close'].rolling(20).std()
        df['RSI14'] = calc_rsi(df['Close'], 14)
        df['RSI2'] = calc_rsi(df['Close'], 2)
        df['Max2'] = df['High'].rolling(2).max()
        df['Donchian_High20'] = df['High'].shift(1).rolling(20).max()
        df['Donchian_Low10'] = df['Low'].shift(1).rolling(10).min()
        df['AvgVolume5'] = df['Volume'].rolling(5).mean()
        
        # RS
        ibov_ret = ibov['Close'].pct_change(21)
        stock_ret = df['Close'].pct_change(21)
        df = df.join(ibov_ret.to_frame('IBOV_Ret21'), how='left')
        df['RS'] = stock_ret - df['IBOV_Ret21']
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        processed_stocks[tkr] = df

    print("\nExecutando Backtest Comparativo V2 (Equal Capital Sizing, Wider Stops)...")
    strategies = ['EMA_Crossover', 'Bollinger_Bands', 'Rsi2_Connors', 'Donchian_Breakout', 'B3_RAVTH']
    
    results = {}
    for name in strategies:
        trades_df, equity_df = run_backtest_v2(processed_stocks, ibov, name)
        total_ret = (equity_df['Total_Equity'].iloc[-1] / 100000.0 - 1.0) * 100
        print(f"Estratégia {name:18} | Retorno Total: {total_ret:6.2f}% | Total Trades: {len(trades_df):4}")
        results[name] = (trades_df, equity_df)

if __name__ == '__main__':
    main()
