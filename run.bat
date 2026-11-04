@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
python -m unittest discover -s tests -q || echo WARNING: some tests failed - the app can still run
python -m first72 serve %*
