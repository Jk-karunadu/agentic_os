@echo off
echo Starting AgentOS...

:: Start Ollama if not already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama server...
    start "" /B "C:\Users\jkkar\AppData\Local\Programs\Ollama\ollama.exe" serve
    timeout /t 6 /nobreak >NUL
) else (
    echo Ollama already running.
)

:: Start FastAPI backend
echo Starting FastAPI backend...
start "AgentOS Backend" /MIN cmd /c ".venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >NUL

:: Start Streamlit frontend
echo Starting Streamlit frontend...
start "AgentOS Frontend" /MIN cmd /c ".venv\Scripts\streamlit.exe run ui/app.py --server.port 8501"
timeout /t 4 /nobreak >NUL

echo.
echo =============================================
echo  AgentOS is running!
echo  Open browser: http://localhost:8501
echo =============================================
start http://localhost:8501
