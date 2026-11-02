@echo off
chcp 65001 >nul
rem ============================================================
rem Plastic Memory - Standalone App Launcher
rem Start the resident service + world service + chat window
rem (Edge app mode, no browser frame)
rem ============================================================
setlocal
set "NODE=C:\Users\wsp\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
set "PLUGIN=C:\Users\wsp\.dsh\plugins\dsh-plastic-memory"
set "SERVER=%PLUGIN%\server\server.mjs"
set "WORLD=%PLUGIN%\server\world-server.mjs"
set "URL=http://127.0.0.1:9630/ui"
set "WORLD_URL=http://127.0.0.1:9631"

rem Start world service (independent program with its own UI)
start "plastic-memory-world" /min "%NODE%" "%WORLD%"

rem Start life service (built-in port-in-use guard)
start "plastic-memory-service" /min "%NODE%" "%SERVER%"

rem Wait for services to be ready
ping -n 3 127.0.0.1 >nul

rem Open the standalone chat window (absolute Edge path, PATH-independent)
set "EDGE1=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
set "EDGE2=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE1%" (
    start "" "%EDGE1%" --app="%URL%" --window-size=1080,760
    goto :done
)
if exist "%EDGE2%" (
    start "" "%EDGE2%" --app="%URL%" --window-size=1080,760
    goto :done
)
start "" "%URL%"

:done
echo [Plastic-Memory] Chat window: %URL%
echo [Plastic-Memory] World UI: %WORLD_URL%
echo [Plastic-Memory] AI model API: http://127.0.0.1:9630/v1/chat/completions (model=digital-life)
endlocal
