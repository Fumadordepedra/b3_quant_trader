import os
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import download_and_filter_data, CACHE_FILE
from strategies import precompute_indicators
from backtester import run_portfolio_backtest
from analyzer import calculate_metrics, run_bootstrap_pvalue, plot_equity_curves

def simulate_ibov_buy_and_hold(ibov_df, initial_capital=100000.0):
    """
    Simula uma estratégia Buy and Hold simples no IBOVESPA (com custos de corretagem/emolumentos).
    """
    ibov = ibov_df.copy().sort_index()
    first_close = ibov['Close'].iloc[0]
    
    SLIPPAGE = 0.001
    EMOLUMENTOS = 0.0003
    
    # Comprar no primeiro dia
    entry_price = first_close * (1 + SLIPPAGE)
    shares = initial_capital / (entry_price * (1 + EMOLUMENTOS))
    
    daily_equity = []
    for date, row in ibov.iterrows():
        current_val = row['Close'] * shares
        daily_equity.append({
            'Date': date,
            'Cash': 0.0,
            'Positions_Value': current_val,
            'Total_Equity': current_val
        })
        
    return pd.DataFrame(daily_equity)

def main():
    print("="*80)
    print(" B3 SWING TRADE QUANTITATIVE COMPARATIVE BACKTESTER (2018 - 2026) ")
    print("="*80)
    
    # 1. Carregar dados históricos
    if not os.path.exists(CACHE_FILE):
        print("Arquivo de cache local não encontrado. Iniciando download...")
        cache = download_and_filter_data()
    else:
        print(f"Carregando dados históricos do cache local em: {CACHE_FILE}")
        cache = pd.read_pickle(CACHE_FILE)
        
    stocks = cache['stocks']
    ibov = cache['ibov']
    
    # 2. Pré-computar indicadores
    processed_stocks, processed_ibov = precompute_indicators(stocks, ibov)
    
    # Estratégias a serem executadas
    STRATEGIES = [
        'EMA_Crossover',
        'Bollinger_Bands',
        'Rsi2_Connors',
        'Donchian_Breakout',
        'B3_RAVTH' # Nossa proposta adaptativa
    ]
    
    all_equities = {}
    all_trades = {}
    
    # 3. Executar o Benchmark: Buy & Hold IBOVESPA
    print("\nExecutando Benchmark: Buy & Hold IBOVESPA...")
    ibov_bh_equity = simulate_ibov_buy_and_hold(processed_ibov)
    all_equities['Buy_Hold_IBOV'] = ibov_bh_equity
    all_trades['Buy_Hold_IBOV'] = pd.DataFrame()  # Sem trades individuais
    
    # 4. Executar cada estratégia no Backtester de Carteira
    for strat in STRATEGIES:
        trades_df, equity_df = run_portfolio_backtest(
            processed_stocks, 
            processed_ibov, 
            strategy_name=strat, 
            initial_capital=100000.0,
            risk_pct=0.01
        )
        all_equities[strat] = equity_df
        all_trades[strat] = trades_df
        print(f"[OK] Estratégia {strat} concluída com {len(trades_df)} trades.")
        
    # 5. Salvar curvas de capital comparativas
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equity_curves_comparison.png')
    plot_equity_curves(all_equities, plot_path, split_date='2023-01-01')
    
    # 6. Calcular e Exibir Métricas Comparativas
    print("\n" + "="*95)
    print(" METRICAS COMPARATIVAS - PERIODO TOTAL (2018 - 2026)")
    print("="*95)
    
    metrics_list = []
    for name, equity_df in all_equities.items():
        trades_df = all_trades[name]
        metrics = calculate_metrics(equity_df, trades_df)
        metrics['Strategy'] = name
        metrics_list.append(metrics)
        
    df_metrics_all = pd.DataFrame(metrics_list).set_index('Strategy')
    print(df_metrics_all[[
        'Total Return (%)', 'CAGR (% a.a.)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)', 'Total Trades'
    ]].to_string(float_format="%.2f"))
    
    # 7. Separar análise: In-Sample (2018-2022) vs Out-of-Sample (2023-2026)
    print("\n" + "="*95)
    print(" METRICAS IN-SAMPLE (2018 - 2022)")
    print("="*95)
    
    is_metrics_list = []
    split_dt = pd.Timestamp('2023-01-01')
    
    for name, equity_df in all_equities.items():
        trades_df = all_trades[name]
        
        # Filtros In-Sample
        equity_is = equity_df[equity_df['Date'] < split_dt]
        if not trades_df.empty:
            trades_is = trades_df[trades_df['Exit Date'] < split_dt]
        else:
            trades_is = trades_df
            
        metrics_is = calculate_metrics(equity_is, trades_is)
        metrics_is['Strategy'] = name
        is_metrics_list.append(metrics_is)
        
    df_metrics_is = pd.DataFrame(is_metrics_list).set_index('Strategy')
    print(df_metrics_is[[
        'Total Return (%)', 'CAGR (% a.a.)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)', 'Total Trades'
    ]].to_string(float_format="%.2f"))
    
    print("\n" + "="*95)
    print(" METRICAS OUT-OF-SAMPLE (2023 - 2026) - VALIDAÇÃO CRÍTICA")
    print("="*95)
    
    oos_metrics_list = []
    for name, equity_df in all_equities.items():
        trades_df = all_trades[name]
        
        # Filtros Out-of-Sample
        equity_oos = equity_df[equity_df['Date'] >= split_dt]
        # Ajustar base 100 para o Out-of-Sample para que mostre o retorno apenas desse período
        if not equity_oos.empty:
            start_val = equity_oos['Total_Equity'].iloc[0]
            equity_oos = equity_oos.copy()
            # Ajustar para o cálculo correto das métricas locais OOS
            equity_oos['Total_Equity'] = (equity_oos['Total_Equity'] / start_val) * 100000.0
            
        if not trades_df.empty:
            trades_oos = trades_df[trades_df['Exit Date'] >= split_dt]
            if not trades_oos.empty:
                trades_oos = trades_oos.copy()
                trades_oos['Net Profit'] = trades_oos['Net Profit'] * (100000.0 / start_val)
        else:
            trades_oos = trades_df
            
        metrics_oos = calculate_metrics(equity_oos, trades_oos)
        metrics_oos['Strategy'] = name
        oos_metrics_list.append(metrics_oos)
        
    df_metrics_oos = pd.DataFrame(oos_metrics_list).set_index('Strategy')
    print(df_metrics_oos[[
        'Total Return (%)', 'CAGR (% a.a.)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)', 'Total Trades'
    ]].to_string(float_format="%.2f"))
    
    # 8. Validação Estatística de Excesso de Retorno da B3-RAVTH
    print("\n" + "="*95)
    print(" VALIDAÇÃO ESTATÍSTICA (BOOTSTRAP MONTE CARLO)")
    print("="*95)
    
    strat_equity = all_equities['B3_RAVTH']
    ibov_equity = all_equities['Buy_Hold_IBOV']
    
    p_val_ibov = run_bootstrap_pvalue(strat_equity, ibov_equity)
    print(f"p-valor do excesso de retorno de B3-RAVTH sobre Buy & Hold IBOVESPA: {p_val_ibov:.4f}")
    if p_val_ibov < 0.05:
        print("[Significativo!] Rejeita-se H0 com 95% de confiança. A superioridade da estratégia não é fruto do acaso.")
    else:
        print("[Não Significativo] Não se pode rejeitar H0 ao nível de 5%. A diferença pode ser fruto da variância.")
        
    # Comparar contra a melhor estratégia de baseline (excluindo IBOV e B3-RAVTH)
    best_baseline = None
    best_baseline_return = -99999
    
    for name in STRATEGIES:
        if name == 'B3_RAVTH':
            continue
        ret = df_metrics_all.loc[name, 'Total Return (%)']
        if ret > best_baseline_return:
            best_baseline_return = ret
            best_baseline = name
            
    if best_baseline:
        baseline_equity = all_equities[best_baseline]
        p_val_base = run_bootstrap_pvalue(strat_equity, baseline_equity)
        print(f"p-valor do excesso de retorno de B3-RAVTH sobre a melhor baseline ({best_baseline}): {p_val_base:.4f}")
        if p_val_base < 0.05:
            print(f"[Significativo!] B3-RAVTH superou estatisticamente a melhor baseline de mercado ({best_baseline}).")
        else:
            print(f"[Não Significativo] B3-RAVTH não superou a baseline {best_baseline} de forma estatisticamente significante.")
            
    # Salvar resultados em CSV para replicação do usuário
    df_metrics_all.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b3_quant_final_results.csv'))
    print("\n[OK] Resultados completos salvos em 'b3_quant_final_results.csv'.")
    print("="*95 + "\n")

if __name__ == '__main__':
    main()
