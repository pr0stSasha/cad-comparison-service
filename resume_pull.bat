@echo off
rem Автоматическое возобновление загрузки модели qwen2.5vl:7b.
rem Скрипт бесконечно перезапускает ollama pull, пока модель не скачается.
rem Работает даже при нестабильном интернете: каждый обрыв = перезапуск через 15 сек.

:loop
echo [%date% %time%] Zapusk zagruzki...
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen2.5vl:7b
if %errorlevel%==0 goto done
echo.
echo ============================================================
echo  Zagruzka prervalas. Perezapusk cherez 15 sekund...
echo  (skachannaya chast sohranena, kachat s nulya ne pridyotsya)
echo ============================================================
timeout /t 15 /nobreak >nul
goto loop

:done
echo.
echo ============================================================
echo  MODELL qwen2.5vl:7b SKACHANA! Mozhno testirovat agenta:
echo  otkroy v brauzere http://localhost:8000
echo ============================================================
pause
