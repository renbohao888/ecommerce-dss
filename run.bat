@echo off
chcp 65001 >nul
title 电子商务大数据分析与智能决策支持系统
cd /d "%~dp0"
python -m pip install -r requirements.txt
echo 启动系统，请在浏览器打开 http://127.0.0.1:5000
python app.py
pause
