@echo off
echo ====================================
echo   Bajaj Finserv AI Chatbot Startup
echo ====================================
echo.

echo 1. Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)
echo ✅ Python is installed

echo.
echo 2. Installing required packages...
echo Installing pandas, numpy, matplotlib, plotly, gradio, PyPDF2...
pip install pandas numpy matplotlib plotly gradio PyPDF2
echo ✅ Packages installed

echo.
echo 3. Checking data files...
if not exist "BFS_Share_Price.csv" (
    echo ❌ ERROR: BFS_Share_Price.csv not found
    echo Please make sure the CSV file is in the same folder as this script
    pause
    exit /b 1
)
echo ✅ CSV file found

echo.
echo 4. Running debug check...
python debug_data.py
echo.

echo 5. Starting the chatbot...
echo The chatbot will start in your browser
echo Press Ctrl+C to stop the chatbot
echo.
python fix_chatbot.py

pause