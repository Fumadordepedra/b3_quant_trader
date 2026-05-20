@echo off
title B3 Swing Trade Quant Scanner
echo ======================================================================
echo          INICIANDO SERVIDOR WEB DO SCANNER QUANTITATIVO B3
echo ======================================================================
echo.
echo [*] Verificando dependencias Python...
python -c "import flask, pandas, numpy, matplotlib, scipy, yfinance" 2>nul
if %errorlevel% neq 0 (
    echo [-] Erro: Dependencias faltando!
    echo Instale rodando: pip install flask pandas numpy matplotlib scipy yfinance
    pause
    exit /b
)
echo [OK] Dependencias encontradas.
echo [*] Inicializando o servidor Flask...
echo.
echo Abra o navegador em: http://127.0.0.1:5000
echo.
echo Pressione Ctrl+C para encerrar o servidor.
echo ======================================================================
python app.py
pause
