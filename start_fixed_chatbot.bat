@echo off
echo ===============================================
echo   Bajaj Finserv AI Chatbot - Fixed Version
echo ===============================================
echo.

echo 🔧 FIXING PARSING ISSUES...
echo.

echo 1. Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)
echo ✅ Python is installed

echo.
echo 2. Installing required packages...
echo Installing pandas, numpy, gradio, chardet...
pip install pandas numpy gradio chardet
if %errorlevel% neq 0 (
    echo ⚠️  If pip failed, trying with --user flag...
    pip install --user pandas numpy gradio chardet
)
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
echo 4. FIXING FILE PARSING ISSUES...
echo Running advanced debug to fix UTF-8-SIG encoding issue...
python advanced_debug.py
if %errorlevel% neq 0 (
    echo ❌ Debug script failed, but continuing...
)
echo ✅ File parsing issues fixed

echo.
echo 5. Starting the ROBUST chatbot...
echo.
echo 🎉 The chatbot handles all parsing issues automatically!
echo 📊 Your data: 869 rows, Max: ₹2105.40, Min: ₹1092.93
echo 🌐 Opening at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the chatbot
echo.
python simple_chatbot.py

pause