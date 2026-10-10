@echo off
setlocal
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

if exist ".venv\Scripts\uvicorn.exe" (
    set UVI=".venv\Scripts\uvicorn.exe"
) else if exist "..\..\clinical-xai-multimodal-depression-fl\.venv\Scripts\uvicorn.exe" (
    set UVI="..\..\clinical-xai-multimodal-depression-fl\.venv\Scripts\uvicorn.exe"
) else (
    set UVI="uvicorn"
)

echo ======================================================================
echo  MENJALANKAN FED-MIND CENTRALIZED APPLICATION SERVER (FASTAPI)
echo ======================================================================
echo Menggunakan: %UVI%
echo Server URL : http://127.0.0.1:8000
echo Swagger UI : http://127.0.0.1:8000/docs
echo ======================================================================
echo.

%UVI% app.main:app --reload --host 127.0.0.1 --port 8000
pause
