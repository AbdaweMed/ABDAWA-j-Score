#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=admin
export SECRET_KEY=change-me-in-prod
python app.py
