import pandas as pd
import numpy as np
from datetime import datetime
from strategies import get_signals_for_day

def is_etf(ticker):
    """Retorna True se o ativo for um ETF (não sujeito à isenção de R$ 20k)."""
    return ticker in ['BOVA11.SA', 'SMAL11.SA']

def run_portfolio_backtest(stocks_data, ibov_df, strategy_name, initial_capital=100000.0, risk_pct=0.01):
    """
    Executa o simulador de carteira real (Portfolio Backtest) multi-ativo dia a dia.
    Implementa dimensionamento igualitário de capital (20% do NAV por trade) e time stop de 8 dias.
    """
    
    # Parâmetros
    SLIPPAGE = 0.001       # 0.1% de slippage na compra e venda
    EMOLUMENTOS = 0.0003   # 0.03% de emolumentos B3
    CDI_ANUAL = 0.10       # 10% a.a. de rendimento do caixa livre (Selic histórica média)
    CDI_DIARIO = (1.0 + CDI_ANUAL) ** (1.0 / 252.0) - 1.0
    
    all_dates = sorted(list(ibov_df.index))
    
    cash = initial_capital
    positions = {}  # ticker -> {entry_date, entry_price, shares, stop_loss, days_held}
    
    trades_log = []
    daily_equity = []
    
    # Controle Fiscal (Imposto de Renda mensal)
    loss_carryforward = 0.0  # Prejuízo acumulado a compensar
    monthly_sales_volume_stocks = 0.0  # Volume total de vendas de ações no mês corrente
    monthly_trades_closed = []  # Lista de trades fechados no mês corrente
    
    current_month = all_dates[0].month
    
    for day_idx in range(1, len(all_dates)):
        current_date = all_dates[day_idx]
        prev_date = all_dates[day_idx - 1]
        
        # 1. Rendimento do caixa livre em CDI no dia de hoje
        cash *= (1 + CDI_DIARIO)
        
        # 2. Fechamento de mês: Calcular Imposto de Renda
        if current_date.month != current_month:
            cash = process_monthly_taxes(cash, monthly_trades_closed, monthly_sales_volume_stocks, loss_carryforward)
            monthly_sales_volume_stocks = 0.0
            monthly_trades_closed = []
            current_month = current_date.month
            
        # 3. Calcular NAV atual
        open_positions_value = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks_data[tkr]
            current_price = df_stock.loc[current_date, 'Close'] if current_date in df_stock.index else pos['entry_price']
            open_positions_value += current_price * pos['shares']
        portfolio_value = cash + open_positions_value
        
        # 4. Verificar Saídas (Vendas)
        tickers_in_position = list(positions.keys())
        for tkr in tickers_in_position:
            pos = positions[tkr]
            df_stock = stocks_data[tkr]
            
            if current_date not in df_stock.index:
                continue
                
            row_stock = df_stock.loc[current_date]
            idx_stock = df_stock.index.get_loc(current_date)
            
            # Buscar sinal de saída da estratégia
            _, is_sell, _, _ = get_signals_for_day(strategy_name, tkr, df_stock, idx_stock)
            
            # Checagem de Stop Loss (Wider Stops: 3.5 * ATR ou 4 * ATR)
            hit_stop = row_stock['Low'] < pos['stop_loss']
            
            exit_trade = False
            exit_price = 0.0
            exit_reason = ""
            
            if hit_stop:
                exit_price = min(pos['stop_loss'], row_stock['Open']) * (1 - SLIPPAGE)
                exit_reason = "Stop Loss"
                exit_trade = True
            elif is_sell:
                exit_price = row_stock['Close'] * (1 - SLIPPAGE)
                exit_reason = "Sinal Saida"
                exit_trade = True
            elif pos['days_held'] >= 8: # Time stop ideal de 8 dias úteis
                exit_price = row_stock['Close'] * (1 - SLIPPAGE)
                exit_reason = "Time Stop"
                exit_trade = True
                
            if exit_trade:
                gross_value = exit_price * pos['shares']
                custos_venda = gross_value * EMOLUMENTOS
                net_value = gross_value - custos_venda
                
                cash += net_value
                
                entry_value = pos['entry_price'] * pos['shares']
                custos_compra = entry_value * EMOLUMENTOS
                net_profit = net_value - (entry_value + custos_compra)
                ret_pct = (exit_price / pos['entry_price'] - 1.0) * 100
                
                trade_record = {
                    'Ticker': tkr,
                    'Entry Date': pos['entry_date'],
                    'Exit Date': current_date,
                    'Days': pos['days_held'],
                    'Entry Price': pos['entry_price'],
                    'Exit Price': exit_price,
                    'Shares': pos['shares'],
                    'Net Profit': net_profit,
                    'Return %': ret_pct,
                    'Reason': exit_reason,
                    'Is_ETF': is_etf(tkr)
                }
                
                trades_log.append(trade_record)
                monthly_trades_closed.append(trade_record)
                
                if not is_etf(tkr):
                    monthly_sales_volume_stocks += gross_value
                    
                del positions[tkr]
            else:
                positions[tkr]['days_held'] += 1
                
        # Recalcular NAV
        open_positions_value = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks_data[tkr]
            current_price = df_stock.loc[current_date, 'Close']
            open_positions_value += current_price * pos['shares']
        portfolio_value = cash + open_positions_value
        
        # 5. Verificar Entradas (Compras)
        max_positions = 5
        available_slots = max_positions - len(positions)
        
        if available_slots > 0 and cash > 2000.0:
            candidates = []
            
            for tkr, df_stock in stocks_data.items():
                if tkr in positions:
                    continue
                    
                if current_date not in df_stock.index:
                    continue
                    
                idx_stock = df_stock.index.get_loc(current_date)
                is_buy, _, stop_loss, ranking_score = get_signals_for_day(strategy_name, tkr, df_stock, idx_stock)
                
                if is_buy and idx_stock > 0:
                    candidates.append({
                        'Ticker': tkr,
                        'Close': df_stock.loc[current_date, 'Close'],
                        'Stop_Loss': stop_loss,
                        'Score': ranking_score
                    })
            
            candidates = sorted(candidates, key=lambda x: x['Score'], reverse=True)
            
            for cand in candidates[:available_slots]:
                entry_price = cand['Close'] * (1 + SLIPPAGE)
                
                # Equal Capital Allocation: 20% do NAV por trade
                pos_value = portfolio_value * 0.20
                if pos_value > cash:
                    pos_value = cash - 50.0
                    
                shares = int(pos_value / entry_price)
                
                if shares >= 1:
                    total_cost = shares * entry_price
                    total_fees = total_cost * EMOLUMENTOS
                    
                    cash -= (total_cost + total_fees)
                    positions[cand['Ticker']] = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'stop_loss': cand['Stop_Loss'],
                        'days_held': 0
                    }
                    
        # Salvar o NAV diário
        open_positions_value = 0.0
        for tkr, pos in positions.items():
            df_stock = stocks_data[tkr]
            current_price = df_stock.loc[current_date, 'Close']
            open_positions_value += current_price * pos['shares']
            
        portfolio_value = cash + open_positions_value
        daily_equity.append({
            'Date': current_date,
            'Cash': cash,
            'Positions_Value': open_positions_value,
            'Total_Equity': portfolio_value
        })
        
    if monthly_trades_closed:
        cash = process_monthly_taxes(cash, monthly_trades_closed, monthly_sales_volume_stocks, loss_carryforward)
        daily_equity[-1]['Cash'] = cash
        daily_equity[-1]['Total_Equity'] = cash + daily_equity[-1]['Positions_Value']
        
    return pd.DataFrame(trades_log), pd.DataFrame(daily_equity)

def process_monthly_taxes(cash, monthly_trades, monthly_sales_volume, loss_carryforward):
    """
    Calcula e debita o Imposto de Renda (15%) conforme legislação da Receita Federal para Swing Trade.
    """
    if not monthly_trades:
        return cash
        
    net_profit_stocks = 0.0
    net_profit_etfs = 0.0
    
    for t in monthly_trades:
        if t['Is_ETF']:
            net_profit_etfs += t['Net Profit']
        else:
            net_profit_stocks += t['Net Profit']
            
    is_stock_exempt = (monthly_sales_volume <= 20000.0)
    
    taxable_stock = 0.0
    if net_profit_stocks > 0:
        if is_stock_exempt:
            taxable_stock = 0.0
        else:
            taxable_stock = net_profit_stocks
    else:
        loss_carryforward += abs(net_profit_stocks)
        
    taxable_etf = 0.0
    if net_profit_etfs > 0:
        taxable_etf = net_profit_etfs
    else:
        loss_carryforward += abs(net_profit_etfs)
        
    total_taxable_profit = taxable_stock + taxable_etf
    
    if total_taxable_profit > 0 and loss_carryforward > 0:
        if loss_carryforward >= total_taxable_profit:
            loss_carryforward -= total_taxable_profit
            total_taxable_profit = 0.0
        else:
            total_taxable_profit -= loss_carryforward
            loss_carryforward = 0.0
            
    if total_taxable_profit > 0:
        imposto = total_taxable_profit * 0.15
        cash -= imposto
        
    return cash
