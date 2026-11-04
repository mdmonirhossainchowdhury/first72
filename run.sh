#!/usr/bin/env bash
# One-command start: creates a venv, installs deps, runs tests, starts the web UI.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python -m unittest discover -s tests -q || echo 'WARNING: some tests failed (the app can still run). See output above.'
python -m first72 serve "$@"
