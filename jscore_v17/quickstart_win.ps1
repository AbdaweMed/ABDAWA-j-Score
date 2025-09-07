python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:ADMIN_USER="admin"
$env:ADMIN_PASS="admin"
$env:SECRET_KEY="change-me-in-prod"
python app.py
