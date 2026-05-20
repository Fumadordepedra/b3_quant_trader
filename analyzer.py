import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime

def calculate_metrics(daily_equity, trades_df, initial_capital=100000.0):
    """
    Calcula todas as métricas essenciais de performance quantitativa
    baseadas na curva de capital diária e no registro de trades.
    """
    if daily_equity.empty:
        return {}
        
    # Rentabilidades
    daily_equity = daily_equity.copy()
    daily_equity.set_index('Date', inplace=True)
    daily_equity['Returns'] = daily_equity['Total_Equity'].pct_change()
    
    total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
    years = total_days / 365.25
    
    total_return = (daily_equity['Total_Equity'].iloc[-1] / initial_capital - 1.0) * 100
    
    # CAGR (Compound Annual Growth Rate)
    cagr = ((daily_equity['Total_Equity'].iloc[-1] / initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100
    
    # Volatilidade Anualizada
    ann_vol = daily_equity['Returns'].std() * np.sqrt(252) * 100
    
    # Sharpe Ratio Anualizado (considerando livre de risco = 0, pois o caixa já rende CDI)
    avg_return_daily = daily_equity['Returns'].mean()
    std_return_daily = daily_equity['Returns'].std()
    sharpe = (avg_return_daily / (std_return_daily + 1e-9)) * np.sqrt(252) if std_return_daily > 0 else 0
    
    # Sortino Ratio Anualizado
    downside_returns = daily_equity['Returns'].clip(upper=0)
    std_downside = downside_returns.std()
    sortino = (avg_return_daily / (std_downside + 1e-9)) * np.sqrt(252) if std_downside > 0 else 0
    
    # Drawdown Máximo
    daily_equity['Peak'] = daily_equity['Total_Equity'].cummax()
    daily_equity['Drawdown'] = (daily_equity['Total_Equity'] - daily_equity['Peak']) / daily_equity['Peak']
    max_dd = daily_equity['Drawdown'].min() * 100
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Métricas de Trades
    total_trades = len(trades_df)
    if total_trades > 0:
        wins = trades_df[trades_df['Net Profit'] > 0]
        losses = trades_df[trades_df['Net Profit'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100
        
        sum_wins = wins['Net Profit'].sum()
        sum_losses = abs(losses['Net Profit'].sum())
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 99.0
        
        avg_win = wins['Net Profit'].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses['Net Profit'].mean()) if len(losses) > 0 else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else 99.0
        
        avg_trade_ret = trades_df['Return %'].mean()
        expectancy = trades_df['Net Profit'].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        payoff = 0.0
        avg_trade_ret = 0.0
        expectancy = 0.0
        
    return {
        'Total Return (%)': total_return,
        'CAGR (% a.a.)': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Calmar Ratio': calmar,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Payoff Ratio': payoff,
        'Avg Trade Return (%)': avg_trade_ret,
        'Expectancy (R$)': expectancy
    }

def run_bootstrap_pvalue(daily_equity_strategy, daily_equity_benchmark):
    """
    Roda um teste de hipótese via Bootstrap de Monte Carlo para calcular
    o p-valor dos retornos diários excedentes da estratégia sobre o benchmark.
    H0: A estratégia não possui retornos superiores ao benchmark (excesso médio <= 0).
    """
    df_strat = daily_equity_strategy.copy().set_index('Date')
    df_bench = daily_equity_benchmark.copy().set_index('Date')
    
    # Sincronizar retornos
    df = df_strat[['Total_Equity']].join(df_bench[['Total_Equity']], lsuffix='_Strat', rsuffix='_Bench')
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    ret_strat = df['Total_Equity_Strat'].pct_change().dropna()
    ret_bench = df['Total_Equity_Bench'].pct_change().dropna()
    
    excess_returns = ret_strat - ret_bench
    obs_mean = excess_returns.mean()
    
    # Se a estratégia perder para o benchmark na média, p-valor é 1.0 (não rejeita H0)
    if obs_mean <= 0:
        return 1.0
        
    # Bootstrap: Centralizar os retornos sob H0 (média = 0)
    centered_excess = excess_returns - obs_mean
    
    # Simulações de Monte Carlo
    n_sims = 1000
    bootstrap_means = []
    
    for _ in range(n_sims):
        sample = np.random.choice(centered_excess, size=len(centered_excess), replace=True)
        bootstrap_means.append(sample.mean())
        
    # p-valor é a fração de simulações com média >= média observada
    p_value = np.sum(np.array(bootstrap_means) >= obs_mean) / n_sims
    return p_value

def plot_equity_curves(all_equities, save_path, split_date='2023-01-01'):
    """
    Plota as curvas de capital de todas as estratégias na mesma figura
    destacando os períodos In-Sample e Out-of-Sample.
    """
    plt.figure(figsize=(14, 7))
    
    for name, df in all_equities.items():
        # Normalizar para Base 100
        normalised_equity = (df['Total_Equity'] / df['Total_Equity'].iloc[0]) * 100
        plt.plot(df['Date'], normalised_equity, label=name, linewidth=2)
        
    # Destacar divisão In-Sample / Out-of-Sample
    dt_split = datetime.strptime(split_date, '%Y-%m-%d')
    plt.axvline(dt_split, color='grey', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Adicionar textos explicativos no gráfico
    plt.text(df['Date'].iloc[int(len(df)*0.25)], 105, 'IN-SAMPLE\n(Treino 2018-2022)', 
             horizontalalignment='center', alpha=0.7, fontweight='bold')
    plt.text(df['Date'].iloc[int(len(df)*0.8)], 105, 'OUT-OF-SAMPLE\n(Validação 2023-2026)', 
             horizontalalignment='center', alpha=0.7, fontweight='bold')
             
    plt.title("Curva de Capital Acumulada Comparativa (Base 100)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Patrimônio Líquido (%)", fontsize=12)
    plt.xlabel("Data", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfico comparativo de curvas de capital salvo em: {save_path}")
