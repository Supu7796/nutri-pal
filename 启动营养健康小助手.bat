@echo off
echo 正在连接 AI 大脑，请稍候...

:: 1. 自动定位 Anaconda 安装路径并激活环境
:: 如果你的 Anaconda 安装在其他盘，请修改下面的路径
set CONDA_PATH=%USERPROFILE%\anaconda3
if not exist "%CONDA_PATH%" set CONDA_PATH=C:\ProgramData\anaconda3

call "%CONDA_PATH%\Scripts\activate.bat" canteen

:: 2. 跳转到项目文件夹
cd /d C:\SmartCanteen

:: 3. 运行主程序
python main.py

pause